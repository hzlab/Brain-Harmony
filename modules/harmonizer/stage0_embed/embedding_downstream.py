import builtins
import os
import sys
from pathlib import Path

import accelerate
import ml_collections
import numpy as np
import torch
from absl import app, flags, logging
from ml_collections import config_flags
from torch.utils.data import DataLoader

from datasets.datasets import get_downstream_fusion_dataset
from modules.harmonizer.util import t1_encoder as t1_encoder_model
from modules.harmonizer.util import utils


def get_pos_embed(device, name, **kwargs):
    import libs.position_embedding as pos_embeds

    return getattr(pos_embeds, name)(device, kwargs["model_args"])


def get_encoder(pos_embed, cls_token, name, **kwargs):
    import libs.model as model

    return getattr(model, name)(pos_embed=pos_embed, cls_token=cls_token, **kwargs)


def train(config):
    if config.get("benchmark", False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f"Process {accelerator.process_index} using device: {device}")

    AbideI_cfgs = [
        config.abide_I_dataset_tr_15,
        config.abide_I_dataset_tr_167,
        config.abide_I_dataset_tr_20,
        config.abide_I_dataset_tr_22,
        config.abide_I_dataset_tr_25,
        config.abide_I_dataset_tr_30,
    ]

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    mini_batch_size = config.train.batch_size  
    total_batch_size = config.train.batch_size * accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.save_root, exist_ok=True)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        utils.set_logger(
            log_level="info", fname=os.path.join(config.workdir, "output.log")
        )
        logging.info(config)
    else:
        utils.set_logger(log_level="error")
        builtins.print = lambda *args: None

    if config.dataset_name == "AbideI":
        train_cfgs = AbideI_cfgs
    else:
        raise NotImplementedError
    train_datasets = [get_downstream_fusion_dataset(**ds_cfg) for ds_cfg in train_cfgs]

    length_set = [len(ds) for ds in train_datasets]
    print("datasets length:")
    print(length_set)

    eval_dataset_loaders = [
        DataLoader(
            train_dataset,
            batch_size=mini_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
            pin_memory=False,
            persistent_workers=True,
        )
        for train_dataset in train_datasets
    ]

    fmri_encoder_pos_embed = get_pos_embed(device, **config.pos_embed)
    fmri_encoder = get_encoder(fmri_encoder_pos_embed, None, **config.encoder)

    ckpt_pth = torch.load(
        config.fmri_pretrain_weights, map_location="cpu", weights_only=True
    )
    prefix = "encoder_ema."
    ema_state_dict = {
        k[len(prefix) :]: v for k, v in ckpt_pth.items() if k.startswith(prefix)
    }
    state_dict = ema_state_dict.copy()
    model_state_dict = fmri_encoder.state_dict()

    for key in list(state_dict.keys()):
        if (
            key in model_state_dict
            and state_dict[key].size() != model_state_dict[key].size()
        ):
            print(
                f"skip param {key} due to size mismatch{state_dict[key].size()} vs {model_state_dict[key].size()}"
            )
            del state_dict[key]

    msg = fmri_encoder.load_state_dict(state_dict, strict=False)
    logging.info(f"load pretrained weights: {msg}")

    t1_encoder = t1_encoder_model.__dict__["mae_vit_base_patch16"](
        img_size=(160, 192, 160)
    )
    state_dict = torch.load(
        config.t1_pretrain_weights, map_location="cpu", weights_only=False
    )["model"]
    msg = t1_encoder.load_state_dict(state_dict, strict=False)
    print(f"load pretrained weights for t1 encoder: {msg}")

    logging.info(f"step_per_epoch before prepare: {len(eval_dataset_loaders[0])}")

    for train_dataset_loaders_i in range(len(eval_dataset_loaders)):
        eval_dataset_loaders[train_dataset_loaders_i] = accelerator.prepare(
            eval_dataset_loaders[train_dataset_loaders_i]
        )
    t1_encoder, fmri_encoder = accelerator.prepare(t1_encoder, fmri_encoder)

    for i in range(len(train_datasets)):
        logging.info(
            f"step_pre_epoch after prepare {i}: {len(eval_dataset_loaders[i])}"
        )

    logging.info(
        f"step_pre_epoch after prepare compare: {len(train_datasets[0]) // total_batch_size}"
    )

    save_dir = config.save_root + config.dataset_name
    os.makedirs(save_dir, exist_ok=True)
    count = 0

    def eval_step(_batch, count):
        with torch.no_grad():
            _batch_fmri = _batch[0]
            _batch_T1 = _batch[1]
            _batch_label = _batch[2]
            _batch_attn_patch_size = _batch[3]
            _batch_attn_attn_mask = _batch[4]
            subject_id = _batch[6]

            fmri = fmri_encoder(
                _batch_fmri,
                _batch_attn_patch_size[0].item(),
                attention_mask=_batch_attn_attn_mask,
            )
            t1 = t1_encoder(_batch_T1)

            x = torch.cat([fmri, t1], dim=1).detach().cpu().numpy()

            tar_path_name = os.path.join(
                save_dir, f"{subject_id[0].replace('/', '~')}.npz"
            )
            count += 1
            np.savez(
                tar_path_name,
                data=x,
                label=_batch_label.detach().cpu().numpy(),
                attn_mask=_batch_attn_attn_mask.detach().cpu().numpy(),
            )
        return count

    logging.info(f"Start generating, mixed_precision={config.mixed_precision}")

    for loader_idx, loader in enumerate(eval_dataset_loaders):
        for _, x_s in enumerate(loader):
            fmri_encoder.eval()
            t1_encoder.eval()

            count = eval_step(x_s, count)
        print(f"count: {count}, dataloader_idx {loader_idx}")


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False
)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith("--config="):
            return Path(argv[i].split("=")[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert "=" in argv[i]
        if argv[i].startswith("--config.") and not argv[i].startswith(
            "--config.dataset.path"
        ):
            hparam, val = argv[i].split("=")
            hparam = hparam.split(".")[-1]
            if hparam.endswith("path"):
                val = Path(val).stem
            lst.append(f"{hparam}={val}")
    hparams = "-".join(lst)
    if hparams == "":
        hparams = "default"
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    config.workdir = FLAGS.workdir or os.path.join(
        "experiments/stage0_embed/downstream_embed"
    )
    train(config)


if __name__ == "__main__":
    app.run(main)

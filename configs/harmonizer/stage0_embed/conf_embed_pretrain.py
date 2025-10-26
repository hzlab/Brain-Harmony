import copy
from dataclasses import dataclass

import ml_collections


@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


VIT_EMBED_DIMS = {
    "vit_small": 384,
    "vit_base": 768,
    "vit_large": 1024,
    "vit_base_flex": 768,
}


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.num_workers = 10
    config.exp_category = "fmri/ukb_abcd_pretrain"

    config.fmri_pretrain_weights = "checkpoints/harmonix-f/model.pth"
    config.t1_pretrain_weights = "checkpoints/harmonix-s/model.pth"

    config.save_root = "experiments/stage0_embed/pretrain_embed/"

    config.use_cls_token = False

    config.eval = False

    config.train = d(
        n_epochs=1,
        batch_size=64,
    )

    # model
    num_patches = 18
    # config.signal_size = (400,48*18)
    config.patch_size = 48
    config.signal_size = (400, config.patch_size * num_patches)
    num_patches_2d = (
        config.signal_size[0],
        num_patches,
    )  # config.signal_size[1]//config.patch_size)

    config.encoder_name = "vit_base_flex"
    config.embed_dim = VIT_EMBED_DIMS[config.encoder_name]

    pos_embed = Args(
        grid_size=num_patches_2d,
        embed_dim=VIT_EMBED_DIMS[config.encoder_name],
        predictor_embed_dim=384,
        cls_token=config.use_cls_token,
        grad_dim=30,
        gradient="brainharmony_pos_embed/gradient_mapping_400.csv",
        geoh_dim=200,
        geo_harm="brainharmony_pos_embed/schaefer400_roi_eigenmodes.csv",
        use_pos_embed_decoder=True,  # False for downstream tasks, True for pretraining
    )

    config.pos_embed = d(
        name="BrainGradient_GeometricHarmonics_Anatomical_400_PosEmbed",
        model_args=pos_embed,
    )

    config.ssl_model = d(
        name="jepa_flex",
    )

    config.encoder = d(
        name=config.encoder_name,
        img_size=config.signal_size,
        patch_size=config.patch_size,
        gradient_checkpointing=False,
    )

    config.ema = (0.996, 1)

    config.decoder = d(
        name="vit_predictor",
        embed_dim=config.embed_dim,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        patch_size=config.patch_size,
    )

    # dataset done
    config.state = "pretrain"
    config.dataset_tr_07 = d(
        name="UKB_fusion",
        state="pretrain",
        split="all",
        n_cortical_rois=400,
        n_subcortical_rois=50,
        seq_length=490,
        statistic_dir="/path/to/statistic_dir/",
        fmri_data_dir="/path/to/fmri data/",
        T1_data_dir="/path/to/T1 data/",
        split_ids_file="/path/to/split_ids.pkl",
        use_subcortical=False,
        preprocess="",
        norm="all_robust_scaling",
        downsample=False,
        sampling_rate=1,
        label_name="",
        labels_file="",
        label_norm_stat_file="",
        standard_time=config.patch_size * 0.735,
        target_num_patches=num_patches,
        return_length=True,
        if_fusion=True,
    )

    config.dataset_tr_14 = copy.deepcopy(config.dataset_tr_07)
    config.dataset_tr_14.downsample = True
    config.dataset_tr_14.sampling_rate = 2

    config.dataset_tr_21 = copy.deepcopy(config.dataset_tr_07)
    config.dataset_tr_21.downsample = True
    config.dataset_tr_21.sampling_rate = 3

    config.dataset_tr_28 = copy.deepcopy(config.dataset_tr_07)
    config.dataset_tr_28.downsample = True
    config.dataset_tr_28.sampling_rate = 4

    return config

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

    config.dataset_name = "AbideI"
    config.save_root = "experiments/stage0_embed/downstream_embed/"

    config.use_cls_token = False

    config.eval = False

    config.train = d(
        n_epochs=1,
        batch_size=1,
    )

    # model
    num_patches = 18
    config.patch_size = 48
    config.signal_size = (400, config.patch_size * num_patches)
    num_patches_2d = (
        config.signal_size[0],
        num_patches,
    ) 

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

    ########################################################################################################################
    #                                                 ABIDE I fMRI dataset test                                            #
    ########################################################################################################################

    config.abide_I_dataset_tr_15 = d(
        name="AbideI",
        split="all",
        fmri_data_dir="/path/to/ABIDE1_fMRI",
        T1_data_dir="/path/to/ABIDE1_T1",
        splits_file="/path/to/data_splits.json",
        tr_file="/path/to/TR.json",  # Add TR file
        use_subcortical=False,
        tr_min=1.5,
        tr_max=1.55,
        standard_time=48 * 0.735,
        target_num_patches=18,
        return_ori_length=True,
    )

    config.abide_I_dataset_tr_167 = copy.deepcopy(config.abide_I_dataset_tr_15)
    config.abide_I_dataset_tr_167.tr_min = 1.6
    config.abide_I_dataset_tr_167.tr_max = 1.7

    config.abide_I_dataset_tr_20 = copy.deepcopy(config.abide_I_dataset_tr_15)
    config.abide_I_dataset_tr_20.tr_min = 2.0
    config.abide_I_dataset_tr_20.tr_max = 2.05

    config.abide_I_dataset_tr_22 = copy.deepcopy(config.abide_I_dataset_tr_15)
    config.abide_I_dataset_tr_22.tr_min = 2.2
    config.abide_I_dataset_tr_22.tr_max = 2.25

    config.abide_I_dataset_tr_25 = copy.deepcopy(config.abide_I_dataset_tr_15)
    config.abide_I_dataset_tr_25.tr_min = 2.5
    config.abide_I_dataset_tr_25.tr_max = 2.55

    config.abide_I_dataset_tr_30 = copy.deepcopy(config.abide_I_dataset_tr_15)
    config.abide_I_dataset_tr_30.tr_min = 3.0
    config.abide_I_dataset_tr_30.tr_max = 3.05

    return config

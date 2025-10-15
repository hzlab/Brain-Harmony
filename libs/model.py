from functools import partial

import torch.nn as nn

from .flex_transformer import (
    FlexVisionTransformer,
)


def vit_base_flex(patch_size=16, **kwargs):
    model = FlexVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


VIT_EMBED_DIMS = {
    "vit_small": 384,
    "vit_base": 768,
    "vit_large": 1024,
}

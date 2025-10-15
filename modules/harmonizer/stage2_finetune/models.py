
from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.flex_transformer import Block


class MLPHead(nn.Module):
    def __init__(
        self, in_features: int, hidden_dim: int, out_features: int, dropout_rate: float
    ):
        super().__init__()
        self.lin1 = nn.Linear(in_features, hidden_dim)
        torch.nn.init.kaiming_uniform_(
            self.lin1.weight, mode="fan_in", nonlinearity="relu"
        )
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.kaiming_uniform_(
            self.lin2.weight, mode="fan_in", nonlinearity="relu"
        )
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.lin3 = nn.Linear(hidden_dim, out_features)
        torch.nn.init.kaiming_uniform_(
            self.lin3.weight, mode="fan_in", nonlinearity="relu"
        )

    def forward(self, x):
        x = self.lin1(x)
        x = self.drop1(x)
        x = F.relu(x)

        x = self.lin2(x)
        x = self.drop2(x)
        x = F.relu(x)

        x = self.lin3(x)
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):

    def __init__(
        self,
        global_pool=False,
        use_mlp=False,
        add_pre_mapping=False,
        num_latent_tokens=128,
        attn_mode="flash_attention_2",
        return_attention=False,
        **kwargs,
    ):
        super(VisionTransformer, self).__init__(**kwargs)
        self.add_pre_mapping = add_pre_mapping
        if self.add_pre_mapping:
            self.pre_map = nn.Linear(768, kwargs["embed_dim"])

        del self.patch_embed

        num_patches = 7200 + 1200

        self.num_latent_tokens = num_latent_tokens
        self.latent_tokens = nn.Parameter(
            torch.randn(self.num_latent_tokens, kwargs["embed_dim"])
        )
        self.enc_latent_token_positional_embedding = nn.Parameter(
            torch.randn(self.num_latent_tokens, kwargs["embed_dim"])
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, kwargs["embed_dim"]), requires_grad=False
        )
        self.return_attention = return_attention

        dpr = [
            x.item()
            for x in torch.linspace(0, kwargs["drop_path_rate"], kwargs["depth"])
        ]
        self.blocks = nn.ModuleList(
            [
                Block(
                    kwargs["embed_dim"],
                    kwargs["num_heads"],
                    kwargs["mlp_ratio"],
                    qkv_bias=True,
                    qk_scale=None,
                    drop_path=dpr[i],
                    norm_layer=kwargs["norm_layer"],
                    attn_mode=attn_mode,
                )
                for i in range(kwargs["depth"])
            ]
        )

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm

        if use_mlp:
            mlp_hidden_size = 768
            mlp_hidden_dropout_prob = 0.0
            self.head = MLPHead(
                in_features=mlp_hidden_size,
                hidden_dim=mlp_hidden_size // 2,
                out_features=1,
                dropout_rate=mlp_hidden_dropout_prob,
            )

    def forward_features(self, x, attn_mask):
        if self.add_pre_mapping:
            x = self.pre_map(x)

        B = x.shape[0]

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        latent_tokens = self.latent_tokens.expand(
            x.shape[0], -1, -1
        )
        latent_tokens = (
            latent_tokens + self.enc_latent_token_positional_embedding
        )

        x = torch.cat([x, latent_tokens], dim=1)

        pad = torch.ones(
            attn_mask.shape[0],
            1200 + latent_tokens.shape[1],
            dtype=attn_mask.dtype,
            device=attn_mask.device,
        )
        pad_0 = torch.ones(
            attn_mask.shape[0], 1, dtype=attn_mask.dtype, device=attn_mask.device
        )
        attn_mask = torch.cat([pad_0, attn_mask, pad], dim=1)

        attn_list = []
        for blk in self.blocks:
            if self.return_attention:
                x, attn = blk(
                    x, attention_mask=attn_mask, return_attention=self.return_attention
                )
                attn_list.append(attn)
            else:
                x = blk(x, attention_mask=attn_mask)

        if self.global_pool:
            x = torch.cat([x[:, :1, :], x[:, -self.num_latent_tokens :]], dim=1)
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = torch.cat([x[:, :1, :], x[:, -self.num_latent_tokens :]], dim=1)
            outcome = x[:, 0]

        if self.return_attention:
            return outcome, attn_list
        return outcome

    def forward(self, x, attn_mask):
        if self.return_attention:
            x, attn_list = self.forward_features(x, attn_mask)
        else:
            x = self.forward_features(x, attn_mask)
        x = self.head(x)
        if self.return_attention:
            return x, attn_list
        return x


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        in_chans=1,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        add_pre_mapping=True,
        **kwargs,
    )
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        in_chans=1,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        add_pre_mapping=False,
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        in_chans=1,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        add_pre_mapping=True,
        **kwargs,
    )
    return model


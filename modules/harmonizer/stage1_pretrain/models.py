
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from libs.flex_transformer import Block
from modules.harmonizer.util.pos_embed import get_1d_sincos_pos_embed_from_grid


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid = np.arange(grid_size, dtype=float)

    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class OneTokRegViT(nn.Module):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        num_latent_tokens=128,
        add_pre_mapping=False,
    ):
        super().__init__()
        self.add_pre_mapping = add_pre_mapping
        if self.add_pre_mapping:
            self.pre_map = nn.Linear(768, embed_dim)

        num_patches = 400 * 18 + 1200

        self.num_latent_tokens = num_latent_tokens
        self.latent_tokens = nn.Parameter(
            torch.randn(self.num_latent_tokens, embed_dim)
        )
        self.enc_latent_token_positional_embedding = nn.Parameter(
            torch.randn(self.num_latent_tokens, embed_dim)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.dec_latent_token_positional_embedding = nn.Parameter(
            torch.randn(1, self.num_latent_tokens, decoder_embed_dim)
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, 768, bias=True
        )

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], 400 * 18 + 1200, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], 400 * 18 + 1200, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        d = imgs.shape[4] // p

        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p, d, p))
        x = torch.einsum("nchpwqdz->nhwdpqzc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p**3 * 1))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = 10
        w = 12
        d = 10

        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p, 1))
        x = torch.einsum("nhwdpqzc->nchpwqdz", x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, w * p, d * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(
            noise, dim=1
        )
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, attn_mask):

        target = x

        if self.add_pre_mapping:
            x = self.pre_map(x)
        x = x + self.pos_embed[:, 1:, :]


        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

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

        for blk in self.blocks:
            x = blk(x, attention_mask=attn_mask)

        x = self.norm(x)

        latent_tokens = torch.cat([x[:, :1, :], x[:, -self.num_latent_tokens :]], dim=1)

        return latent_tokens, target

    def forward_decoder(self, x):
        B, N, C = x.shape
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], 400 * 18 + 1200, 1)

        x_ = torch.cat([x[:, :1, :], mask_tokens], dim=1) + self.decoder_pos_embed

        x = torch.cat(
            [x_, x[:, 1:, :] + self.dec_latent_token_positional_embedding], dim=1
        )

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1 : mask_tokens.shape[1] + 1, :]
        return x

    def forward_loss(self, imgs, pred):

        target = imgs

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean()
        return loss

    def forward(self, imgs, attn_mask):
        latent, target = self.forward_encoder(imgs, attn_mask)
        pred = self.forward_decoder(latent)
        loss = self.forward_loss(target, pred)
        return loss, pred, None


def onetokreg_vit_base_patch16_dec512d8b(**kwargs):
    model = OneTokRegViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        add_pre_mapping=False,
        **kwargs,
    )
    return model


onetokreg_vit_base_patch16 = onetokreg_vit_base_patch16_dec512d8b

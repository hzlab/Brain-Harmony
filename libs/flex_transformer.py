
import logging
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from timm.models.layers import to_3tuple

from libs.flex_patch_embed import FlexiPatchEmbed
from libs.masks.utils import apply_masks

from .attn_utils.fa2_utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
)
from .utils import repeat_interleave_batch, trunc_normal_

logger = logging.getLogger(__name__)


if is_flash_attn_2_available():
    from .attn_utils.modeling_flash_attention_utils import _flash_attention_forward


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.dropout = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attention_mask, output_attentions=False):
        if attention_mask is not None:
            from attn_utils import expand_mask

            attention_mask = expand_mask(attention_mask, x.dtype)

        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        src_len = k.size(2)

        if attn.size() != (B, self.num_heads, N, src_len):
            raise ValueError(
                f"Attention weights should be of size {(B, self.num_heads, N, src_len)}, but is"
                f" {attn.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (B, 1, N, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(B, 1, N, src_len)}, but is {attention_mask.size()}"
                )
            breakpoint()
            attn = attn + attention_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class FlashAttention2(Attention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        output_attentions = False

        B, N, C = hidden_states.shape

        qkv = self.qkv(hidden_states).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        )
        query_states, key_states, value_states = (
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
        )


        dropout_rate = self.dropout if self.training else 0.0


        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            N,
            dropout=dropout_rate,
            is_causal=causal_attention_mask is not None,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(B, N, C).contiguous()

        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


CLIP_ATTENTION_CLASSES = {
    "eager": Attention,
    "flash_attention_2": FlashAttention2,
}


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_mode="flash_attention_2",
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = CLIP_ATTENTION_CLASSES[attn_mode](
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, attention_mask=None, return_attention=False):
        y, attn = self.attn(
            self.norm1(x),
            attention_mask=attention_mask,
            output_attentions=return_attention,
        )

        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=(450, 490), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[0]) * (img_size[1] // patch_size)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_shape = (img_size[0], img_size[1] // patch_size)

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_3D(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        num_patches = (
            (img_size[0] // patch_size[0])
            * (img_size[1] // patch_size[1])
            * (img_size[2] // patch_size[2])
        )
        self.patch_shape = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert (
            D == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2]
        ), (
            f"Input image size ({D}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        )
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformerPredictor(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        patch_size=None,
        pos_embed=None,
        cls_token=None,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        attn_mode="flash_attention_2",
        **kwargs,
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]

        self.cls_token = cls_token
        self.predictor_pos_embed = pos_embed
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    attn_mode=attn_mode,
                )
                for i in range(depth)
            ]
        )

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, masks, attention_masks=None, return_attention=False):
        assert (masks is not None) and (masks_x is not None), (
            "Cannot run predictor without mask indices"
        )

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        B = len(x) // len(masks_x)

        x = self.predictor_embed(x)

        predictor_pos_embed = self.predictor_pos_embed()[1]

        if self.cls_token is not None:
            x_pos_embed = predictor_pos_embed.repeat(B, 1, 1)

            x_pos_embed = apply_masks(x_pos_embed, masks_x, cls_token=True)
            x += x_pos_embed

            _, N_ctxt, D = x.shape

            pos_embs = predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs[:, 1:, :], masks)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))

            pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)

            pred_tokens += pos_embs
        else:
            x_pos_embed = predictor_pos_embed.repeat(B, 1, 1)
            x += apply_masks(x_pos_embed, masks_x)

            _, N_ctxt, D = x.shape

            pos_embs = predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
            pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
            pred_tokens += pos_embs

        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)
        attn_set = []
        for blk in self.predictor_blocks:
            if return_attention:
                x, attn = blk(x, return_attention)
                attn_set.append(attn.detach().cpu())
            else:
                x = blk(x, attention_masks)
        x = self.predictor_norm(x)

        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        if return_attention:
            return x, attn_set
        return x


class VisionTransformerDecoder(VisionTransformerPredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if len(self.predictor_pos_embed.grid_size) == 3:
            self.predictor_proj = nn.Linear(
                kwargs["predictor_embed_dim"], kwargs["patch_size"] ** 3, bias=True
            )
        else:
            self.predictor_proj = nn.Linear(
                kwargs["predictor_embed_dim"], kwargs["patch_size"], bias=True
            )

        trunc_normal_(self.predictor_proj.weight, std=self.init_std)
        nn.init.constant_(self.predictor_proj.bias, 0)


class FlexVisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        pos_embed=None,
        cls_token=None,
        img_size=(224, 224),
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        gradient_checkpointing=False,
        attn_mode="flash_attention_2",
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing
        self.patch_embed = FlexiPatchEmbed(patch_size, in_chans, embed_dim)
        self.cls_token = cls_token
        self.pos_embed = pos_embed
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    attn_mode=attn_mode,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=self.init_std)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self, x, patch_size, masks=None, attention_mask=None, return_attention=False
    ):
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        x = self.patch_embed(x, patch_size)
        B, N, D = x.shape

        pos_embed = self.pos_embed()[0]
        if self.cls_token is not None:
            x = x + pos_embed[:, 1:, :]

            if masks is not None:
                x = apply_masks(x, masks)

            cls_token = self.cls_token + pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            x = x + pos_embed
            if masks is not None:
                x = apply_masks(x, masks)

        attn_set = []
        for i, blk in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                if return_attention:
                    x, attn = torch.utils.checkpoint.checkpoint(
                        blk, x, return_attention, use_reentrant=False
                    )
                    attn_set.append(attn.detach().cpu())
                else:
                    x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                if return_attention:
                    x, attn = blk(x, return_attention)
                    attn_set.append(attn.detach().cpu())
                else:
                    x = blk(x, attention_mask)

        if self.norm is not None:
            x = self.norm(x)

        if return_attention:
            return x, attn_set
        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=math.sqrt(npatch / N),
            mode="bicubic",
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)


class Flex_VisionTransformerForClassification(FlexVisionTransformer):
    """Vision Transformer for Classification Tasks.

    This model extends the base Vision Transformer for classification tasks by adding:
    1. A classification head
    2. Support for global pooling
    3. Flexible handling of CLS token vs global pooling

    Args:
        num_classes (int): Number of classes for classification
        global_pool (bool): Whether to use global average pooling instead of CLS token
        **kwargs: Additional arguments passed to VisionTransformer
    """

    def __init__(self, num_classes: int = 2, global_pool: bool = False, **kwargs):
        super().__init__(**kwargs)

        if global_pool and self.cls_token is not None:
            logger.warning(
                "Using global pooling with CLS token present - CLS token will be ignored for pooling"
            )
        elif not global_pool and self.cls_token is None:
            raise ValueError("Must use global pooling when no CLS token is present")

        self.global_pool = global_pool

        self.fc_norm = kwargs.get("norm_layer", nn.LayerNorm)(
            self.embed_dim
        )
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self._init_weights(self.fc_norm)
        self._init_weights(self.head)

    def forward_features(
        self, x, patch_size, masks: Optional[torch.Tensor] = None, attention_masks=None
    ) -> torch.Tensor:
        """Extract features before classification head.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            masks (Optional[torch.Tensor]): Optional attention masks

        Returns:
            torch.Tensor: Features of shape (B, embed_dim)
        """
        features = super().forward(x, patch_size, masks, attention_masks)

        if self.global_pool:
            if self.cls_token is not None:
                features = features[:, 1:].mean(dim=1)
            else:
                mask = attention_masks.float().unsqueeze(-1) 
                masked_features = features * mask 
                valid_counts = mask.sum(dim=1)
                valid_counts = valid_counts.clamp(min=1)
                features = masked_features.sum(dim=1) / valid_counts
        else:
            features = features[:, 0]

        features = self.fc_norm(features)
        return features

    def forward(
        self,
        x: torch.Tensor,
        patch_size=None,
        attention_masks=None,
        masks: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for classification.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            masks (Optional[torch.Tensor]): Optional attention masks
            return_features (bool): If True, return both features and logits

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - If return_features=False: Classification logits of shape (B, num_classes)
                - If return_features=True: Tuple of (features, logits)
        """
        features = self.forward_features(x, patch_size, masks, attention_masks)
        logits = self.head(features)

        if return_features:
            return features, logits
        return logits

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

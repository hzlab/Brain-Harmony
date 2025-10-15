import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.masks.utils import apply_masks
from libs.utils import repeat_interleave_batch


def get_model(pos_embed, cls_token, name, **kwargs):
    import libs.model as model

    return getattr(model, name)(pos_embed=pos_embed, cls_token=cls_token, **kwargs)


def get_pos_embed(device, name, **kwargs):
    import libs.position_embedding as pos_embeds

    return getattr(pos_embeds, name)(device, kwargs["model_args"])


class JEPA_TS_FLEX(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        # Initialize position embedding
        pos_embed = (
            get_pos_embed(device, **config.pos_embed)
            if config.get("pos_embed", False)
            else None
        )

        # Initialize cls token
        if config.get("use_cls_token", False):
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            cls_token_ema = nn.Parameter(self.cls_token.data.clone())
        else:
            self.cls_token = None
            cls_token_ema = None

        # Initialize encoder and decoder
        self.encoder = get_model(pos_embed, self.cls_token, **config.encoder)
        self.decoder = get_model(pos_embed, self.cls_token, **config.decoder)

        # Initialize encoder_ema
        pos_embed_ema = copy.deepcopy(pos_embed)
        self.encoder_ema = get_model(pos_embed_ema, cls_token_ema, **config.encoder)

        for param in self.encoder_ema.parameters():
            param.requires_grad = False
        self.encoder_ema.eval()

    def forward(
        self,
        signals,
        patch_size,
        masks_enc,
        masks_pred,
        attn_mask_enc,
        attn_mask_dec,
        attn_mask_ema=None,
        return_attention=False,
    ):
        z = self.encoder(
            signals,
            patch_size,
            masks_enc,
            attn_mask_enc,
            return_attention=return_attention,
        )
        z = self.decoder(
            z,
            masks_enc,
            masks_pred,
            torch.cat([attn_mask_enc, attn_mask_dec], dim=1),
            return_attention=return_attention,
        )

        attn_mask_dec_expanded = attn_mask_dec.unsqueeze(-1)
        z = z * attn_mask_dec_expanded
        # breakpoint()
        with torch.no_grad():
            h = self.encoder_ema(signals, patch_size, attention_mask=attn_mask_ema)
            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            B = len(h)
            # -- create targets (masked regions of h)
            h = apply_masks(h, masks_pred)
            h = h * attn_mask_dec_expanded
            h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
        # assert ((z == 0).sum(dim=1) == (h == 0).sum(dim=1)).all().item() may have some generated 0s
        # try:
        #     assert ((z == 0).sum(dim=1) == (h == 0).sum(dim=1)).all().item()
        # except AssertionError:
        #     breakpoint()
        loss = F.smooth_l1_loss(z, h, reduction="none")  # shape: (B, L, D)

        token_loss = loss.mean(dim=-1)  # shape: (B, L)

        if attn_mask_dec.all():
            loss = token_loss.mean(dim=-1)
        else:
            masked_loss = token_loss * attn_mask_dec  # shape: (B, L)
            valid_counts = attn_mask_dec.sum(dim=1)  # shape: (B,)
            loss = masked_loss.sum(dim=1) / valid_counts  # 每个 sample 的平均 loss

        # loss = F.smooth_l1_loss(z, h, reduction='none')
        # loss = loss.flatten(start_dim=1).mean(dim=-1)

        return loss, {"loss": loss}

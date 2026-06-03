"""BLO-SAM extension for instance segmentation.

Adds a Cellpose-style 3-channel (dy, dx, cellprob) flow head on top of the
LoRA-adapted SAM mask decoder. Bilevel role assignment is unchanged:
  - prompt embedding (no_mask_embed) = meta (upper-level, optimized on D2)
  - LoRA layers + new flow head        = non-meta (lower-level, optimized on D1)

`Prompt._get_model_parameters` (in prompt.py) filters by the absence of
"no_mask_embed" in the name, so the flow_head parameters are automatically
included in the non-meta group without any change to prompt.py.

The flow head consumes the SAM image embedding (256x64x64) and the SAM
semantic low-res logits (2x64x64), and produces (dy, dx, prob_logit) at 64x64.
Post-processing at inference uses cellpose.dynamics.compute_masks to turn
the flow field into instance labels.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# Re-use everything from the existing module: _LoRA_qkv, _LoRA_qkv_proj, and the
# core LoRA_Sam class. We subclass it and bolt on the flow head.
from sam_lora_mask_decoder import LoRA_Sam as _LoRA_Sam_Semantic


def _conv_block(ch_in, ch_out, gn_groups=8):
    """A small residual conv block with GroupNorm + GELU."""
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, 3, padding=1),
        nn.GroupNorm(min(gn_groups, ch_out), ch_out),
        nn.GELU(),
        nn.Conv2d(ch_out, ch_out, 3, padding=1),
        nn.GroupNorm(min(gn_groups, ch_out), ch_out),
        nn.GELU(),
    )


class FlowHead(nn.Module):
    """Cellpose-style UNet decoder producing (dy, dx, cell_prob_logit) at 256x256.

    Architecture (4 upsampling stages, 16x16 -> 32 -> 64 -> 128 -> 256):
      - Stage 0: image_embed (256ch @ 16) -> conv block to hidden
      - Stage 1: upsample 32x32, conv block
      - Stage 2: upsample 64x64, CONCAT with sem_logits (semantic prior), conv block
      - Stage 3: upsample 128x128, conv block
      - Stage 4: upsample 256x256, conv block -> output head

    Total parameters: ~1.5M with hidden=128 (vs. ~50K for the shallow version).
    """
    def __init__(self, embed_ch=256, sem_ch=2, hidden=128):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.in_proj = nn.Conv2d(embed_ch, hidden, 1)
        self.stage1 = _conv_block(hidden, hidden)              # 32x32
        self.stage2 = _conv_block(hidden + sem_ch, hidden)     # 64x64, with semantic concat
        self.stage3 = _conv_block(hidden, hidden // 2)         # 128x128
        self.stage4 = _conv_block(hidden // 2, hidden // 2)    # 256x256
        self.out_head = nn.Conv2d(hidden // 2, 3, 1)
        # Learnable per-channel output scale; dy/dx GT have magnitudes ~1.
        self.flow_scale = nn.Parameter(torch.tensor([5.0, 5.0, 1.0]).view(1, 3, 1, 1))

    def forward(self, image_embed, sem_logits):
        # image_embed: (B, 256, 16, 16); sem_logits: (B, 2, 64, 64)
        x = self.in_proj(image_embed)              # 16x16, hidden
        x = self.up(x)                              # 32x32
        x = self.stage1(x)
        x = self.up(x)                              # 64x64
        # concat semantic prior at this scale (sem_logits is also 64x64)
        if sem_logits.shape[-2:] != x.shape[-2:]:
            sem_logits = F.interpolate(sem_logits, size=x.shape[-2:],
                                       mode="bilinear", align_corners=False)
        x = torch.cat([x, sem_logits], dim=1)
        x = self.stage2(x)
        x = self.up(x)                              # 128x128
        x = self.stage3(x)
        x = self.up(x)                              # 256x256
        x = self.stage4(x)
        return self.out_head(x) * self.flow_scale   # (B, 3, 256, 256)


class LoRA_Sam(_LoRA_Sam_Semantic):
    """BLO-SAM with a Cellpose-style flow head for instance segmentation.

    Uses the same name as the parent (LoRA_Sam) so train.py's
    `pkg = import_module(args.module); net = pkg.LoRA_Sam(sam, args.rank)`
    works unchanged when --module sam_lora_mask_decoder_instance.
    """
    def __init__(self, sam_model, r, lora_layer=None, embed_ch=256, sem_ch=2):
        super().__init__(sam_model, r, lora_layer=lora_layer)
        self.flow_head = FlowHead(embed_ch=embed_ch, sem_ch=sem_ch)

    def forward(self, batched_input, multimask_output, image_size):
        """Replicates Sam.forward_train but also returns flow_logits."""
        sam = self.sam
        # tensor input path (same as Sam.forward_train in this codebase)
        input_images = sam.preprocess(batched_input)
        image_embeddings = sam.image_encoder(input_images)               # (B, 256, 64, 64)
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        masks = sam.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size),
        )
        # New head: predicts (dy, dx, cellprob_logit) at low_res
        flow_logits = self.flow_head(image_embeddings, low_res_masks)
        return {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
            "flow_logits": flow_logits,
        }

    # Inherits save_lora_parameters / load_lora_parameters from the parent.
    # The flow_head parameters live under "flow_head.*" in state_dict; we extend save/load:
    def save_lora_parameters(self, filename: str) -> None:
        super().save_lora_parameters(filename)
        # Append flow_head params to the same checkpoint
        ckpt = torch.load(filename, map_location="cpu")
        for k, v in self.flow_head.state_dict().items():
            ckpt[f"flow_head.{k}"] = v
        torch.save(ckpt, filename)

    def load_lora_parameters(self, filename: str, device: torch.device) -> None:
        super().load_lora_parameters(filename, device)
        sd = torch.load(filename, map_location=device)
        flow_sd = {k[len("flow_head."):]: v for k, v in sd.items() if k.startswith("flow_head.")}
        if flow_sd:
            self.flow_head.load_state_dict(flow_sd)

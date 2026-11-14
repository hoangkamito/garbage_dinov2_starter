"""
DINOv2 (frozen backbone) + Deformable DETR head — pure PyTorch implementation.

Vì sao pure PyTorch (không dùng CUDA op deformable_attn gốc)?
  - Deformable DETR bản chính thức cần compile 1 CUDA extension (MultiScaleDeformableAttention),
    rất hay lỗi trên Windows (cần MSVC + nvcc khớp version).
  - Bản này dùng grid_sample (có sẵn trong PyTorch) để mô phỏng deformable attention:
    vẫn giữ đúng ý tưởng "sparse sampling points học được" của paper, chỉ khác về tốc độ
    (chậm hơn CUDA op chuyên dụng nhưng chạy được trên mọi máy, kể cả Windows/CPU debug).

Kiến trúc tổng quan:
    Ảnh
      -> DINOv2 (frozen, ViT-S/14) trích feature ở 4 layer khác nhau
      -> Neck: project + resize thành 4 scale feature map (giả lập FPN)
      -> Deformable Encoder (self-attn dạng deformable trên multi-scale feature)
      -> Deformable Decoder (N object query, cross-attn dạng deformable + self-attn)
      -> Head: MLP -> class logits (5 class) + bbox (cx, cy, w, h)

Cách chạy training (ví dụ):
    python dinov2_deformable_detr.py \
        --coco_json WasteDetect_Dataset/train/annotations_coco.json \
        --images_dir WasteDetect_Dataset/train/images \
        --val_coco_json WasteDetect_Dataset/valid/annotations_coco.json \
        --val_images_dir WasteDetect_Dataset/valid/images \
        --epochs 30 --batch_size 4 --img_size 518
"""

import os
import json
import math
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from scipy.optimize import linear_sum_assignment


# ============================================================
# 1. DINOv2 BACKBONE (frozen) + NECK tạo multi-scale feature
# ============================================================

class DINOv2MultiScaleBackbone(nn.Module):
    """
    Lấy feature từ 4 layer khác nhau của DINOv2 ViT-S/14 (mimic multi-scale của CNN).
    ViT vốn chỉ ra 1 resolution duy nhất (H/14 x W/14) ở mọi layer, nên phải tự tạo
    pyramid bằng cách downsample/upsample các layer feature khác nhau -> 4 scale giả lập.
    """

    LAYER_TAPS = [2, 5, 8, 11]  # lấy feature sau các block này (ViT-S/14 có 12 block, index 0-11)

    def __init__(self, d_model: int = 256, freeze: bool = True):
        super().__init__()
        # torch.hub tải DINOv2 pretrained từ Meta (cần internet lần đầu)
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.patch_size = 14
        self.embed_dim = self.dinov2.embed_dim  # 384 với ViT-S

        if freeze:
            for p in self.dinov2.parameters():
                p.requires_grad = False
            self.dinov2.eval()
        self.freeze = freeze

        # Project 384 -> d_model cho từng scale
        self.input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(self.embed_dim, d_model, kernel_size=1), nn.GroupNorm(32, d_model))
            for _ in self.LAYER_TAPS
        ])

        # Tạo thêm 1 scale downsample nữa từ scale cuối (giống Deformable DETR có level "extra")
        self.extra_downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, d_model),
        )

        # Điều chỉnh resolution tương đối giữa các scale (giả lập FPN pyramid 1x, 1/2x, 1/4x, 1/8x)
        self.scale_resize = [2.0, 1.0, 0.5, 0.25]  # so với resolution gốc H/14 x W/14

    def forward(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        """pixel_values: (B, 3, H, W), H,W chia hết cho 14."""
        B, _, H, W = pixel_values.shape
        h_p, w_p = H // self.patch_size, W // self.patch_size

        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            feats = self.dinov2.get_intermediate_layers(
                pixel_values, n=self.LAYER_TAPS, reshape=False, return_class_token=False
            )
            # feats: tuple gồm len(LAYER_TAPS) tensor, mỗi tensor (B, N_patches, embed_dim)

        multiscale_feats = []
        for i, feat in enumerate(feats):
            # (B, N, C) -> (B, C, h_p, w_p)
            feat_map = feat.transpose(1, 2).reshape(B, self.embed_dim, h_p, w_p)
            feat_map = self.input_proj[i](feat_map)

            # resize theo scale mong muốn để tạo pyramid
            scale = self.scale_resize[i]
            if scale != 1.0:
                new_size = (max(1, int(h_p * scale)), max(1, int(w_p * scale)))
                feat_map = F.interpolate(feat_map, size=new_size, mode="bilinear", align_corners=False)
            multiscale_feats.append(feat_map)

        # thêm 1 level cực nhỏ (downsample thêm từ level cuối)
        extra = self.extra_downsample(multiscale_feats[-1])
        multiscale_feats.append(extra)

        return multiscale_feats  # list 5 tensor, mỗi cái (B, d_model, Hi, Wi)


# ============================================================
# 2. DEFORMABLE ATTENTION (pure PyTorch, dùng grid_sample)
# ============================================================

class MSDeformAttn(nn.Module):
    """
    Multi-Scale Deformable Attention (mô phỏng, dùng grid_sample thay CUDA kernel gốc).
    Ý tưởng: mỗi query chỉ "nhìn" vào K điểm sampling học được (offset) trên mỗi scale,
    thay vì attend toàn bộ feature map như attention thường -> giảm chi phí tính toán,
    đây chính là điểm cốt lõi giúp Deformable DETR hội tụ nhanh hơn DETR gốc.
    """

    def __init__(self, d_model=256, n_heads=8, n_levels=5, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, query, reference_points, value_flatten, spatial_shapes):
        """
        query: (B, Lq, C)
        reference_points: (B, Lq, n_levels, 2)  -- normalized [0,1] toạ độ (x,y) trên mỗi level
        value_flatten: (B, sum(Hi*Wi), C)
        spatial_shapes: list [(H0,W0), (H1,W1), ...]
        """
        B, Lq, C = query.shape
        value = self.value_proj(value_flatten)
        value = value.view(B, -1, self.n_heads, self.head_dim)

        offsets = self.sampling_offsets(query).view(B, Lq, self.n_heads, self.n_levels, self.n_points, 2)
        attn_w = self.attention_weights(query).view(B, Lq, self.n_heads, self.n_levels * self.n_points)
        attn_w = F.softmax(attn_w, dim=-1).view(B, Lq, self.n_heads, self.n_levels, self.n_points)

        # tách value theo từng level để dùng grid_sample (cần dạng ảnh (B*heads, head_dim, H, W))
        split_sizes = [h * w for h, w in spatial_shapes]
        value_list = value.split(split_sizes, dim=1)

        sampled_outputs = []
        offset_norm = torch.tensor([[w, h] for h, w in spatial_shapes], device=query.device, dtype=query.dtype)

        for lvl, (H, W) in enumerate(spatial_shapes):
            # (B, H*W, heads, head_dim) -> (B*heads, head_dim, H, W)
            v_l = value_list[lvl].permute(0, 2, 3, 1).reshape(B * self.n_heads, self.head_dim, H, W)

            # toạ độ sampling = reference_point + offset (normalize theo w,h của level)
            ref_l = reference_points[:, :, lvl, :]  # (B, Lq, 2)
            off_l = offsets[:, :, :, lvl, :, :] / offset_norm[lvl]  # (B, Lq, heads, points, 2)
            sample_loc = ref_l[:, :, None, None, :] + off_l  # (B, Lq, heads, points, 2), normalized [0,1]
            sample_grid = 2 * sample_loc - 1  # grid_sample cần range [-1,1]

            # reshape để grid_sample: input (B*heads, C, H, W), grid (B*heads, Lq, points, 2)
            grid = sample_grid.permute(0, 2, 1, 3, 4).reshape(B * self.n_heads, Lq, self.n_points, 2)
            sampled = F.grid_sample(v_l, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            # sampled: (B*heads, head_dim, Lq, points)
            sampled = sampled.view(B, self.n_heads, self.head_dim, Lq, self.n_points)
            sampled_outputs.append(sampled)

        sampled_outputs = torch.stack(sampled_outputs, dim=-2)  # (B, heads, head_dim, Lq, levels, points)
        attn_w = attn_w.permute(0, 2, 1, 3, 4)  # (B, heads, Lq, levels, points)
        out = (sampled_outputs * attn_w.unsqueeze(2)).sum(dim=(-1, -2))  # (B, heads, head_dim, Lq)
        out = out.permute(0, 3, 1, 2).reshape(B, Lq, C)

        return self.output_proj(out)


def flatten_multiscale(feats: List[torch.Tensor]):
    """(list Bx C x Hi x Wi) -> (B, sum(Hi*Wi), C) + spatial_shapes."""
    B, C = feats[0].shape[0], feats[0].shape[1]
    spatial_shapes = [(f.shape[2], f.shape[3]) for f in feats]
    flat = torch.cat([f.flatten(2).transpose(1, 2) for f in feats], dim=1)  # (B, sum(HW), C)
    return flat, spatial_shapes


# ============================================================
# 3. ENCODER / DECODER LAYER
# ============================================================

class DeformableEncoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=5, n_points=4, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_heads, n_levels, n_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, reference_points, spatial_shapes, pos_embed):
        q = src + pos_embed
        attn_out = self.self_attn(q, reference_points, src, spatial_shapes)
        src = self.norm1(src + self.dropout(attn_out))
        src = self.norm2(src + self.dropout(self.ffn(src)))
        return src


class DeformableDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=5, n_points=4, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformAttn(d_model, n_heads, n_levels, n_points)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, query_pos, reference_points, memory, spatial_shapes):
        q = k = tgt + query_pos
        sa_out, _ = self.self_attn(q, k, tgt)
        tgt = self.norm1(tgt + self.dropout(sa_out))

        ca_out = self.cross_attn(tgt + query_pos, reference_points, memory, spatial_shapes)
        tgt = self.norm2(tgt + self.dropout(ca_out))

        tgt = self.norm3(tgt + self.dropout(self.ffn(tgt)))
        return tgt


# ============================================================
# 4. FULL MODEL: DINOv2 backbone + Deformable DETR head
# ============================================================

class DINOv2DeformableDETR(nn.Module):
    def __init__(self, num_classes=5, d_model=256, n_heads=8, n_levels=5, n_points=4,
                 num_encoder_layers=3, num_decoder_layers=3, num_queries=100, freeze_backbone=True):
        super().__init__()
        self.backbone = DINOv2MultiScaleBackbone(d_model=d_model, freeze=freeze_backbone)
        self.n_levels = n_levels
        self.d_model = d_model

        self.level_embed = nn.Parameter(torch.randn(n_levels, d_model) * 0.02)

        self.encoder_layers = nn.ModuleList([
            DeformableEncoderLayer(d_model, n_heads, n_levels, n_points) for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DeformableDecoderLayer(d_model, n_heads, n_levels, n_points) for _ in range(num_decoder_layers)
        ])

        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)       # content embedding
        self.query_pos_embed = nn.Embedding(num_queries, d_model)   # positional embedding
        self.reference_points_head = nn.Linear(d_model, 2)          # query -> initial reference point (x,y)

        # prediction head (áp dụng ở output mỗi decoder layer để có deep supervision)
        self.class_head = nn.Linear(d_model, num_classes)
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),  # cx, cy, w, h
        )

    def get_pos_embed_2d(self, feat: torch.Tensor) -> torch.Tensor:
        """Sinusoidal 2D positional embedding cho 1 feature map (B,C,H,W) -> (B,H*W,C)."""
        B, C, H, W = feat.shape
        device = feat.device
        y_embed = torch.arange(H, device=device).float().unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, device=device).float().unsqueeze(0).repeat(H, 1)
        dim_t = torch.arange(C // 2, device=device).float()
        dim_t = 10000 ** (2 * (dim_t // 2) / (C // 2))

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
        pos = torch.cat([pos_y, pos_x], dim=-1).view(H * W, -1)  # (H*W, C)
        return pos.unsqueeze(0).repeat(B, 1, 1)

    def forward(self, pixel_values: torch.Tensor):
        feats = self.backbone(pixel_values)  # list 5 x (B, d_model, Hi, Wi)
        B = pixel_values.shape[0]

        pos_embeds = [self.get_pos_embed_2d(f) for f in feats]
        pos_embeds = [p + self.level_embed[i] for i, p in enumerate(pos_embeds)]
        pos_flat = torch.cat(pos_embeds, dim=1)  # (B, sum(HW), C)

        src_flat, spatial_shapes = flatten_multiscale(feats)

        # reference points cho encoder: chính là toạ độ trung tâm mỗi pixel, lặp lại cho mọi level (self-attn)
        ref_points_list = []
        for (H, W) in spatial_shapes:
            ys, xs = torch.meshgrid(
                torch.linspace(0.5 / H, 1 - 0.5 / H, H, device=pixel_values.device),
                torch.linspace(0.5 / W, 1 - 0.5 / W, W, device=pixel_values.device),
                indexing="ij",
            )
            ref = torch.stack([xs, ys], dim=-1).reshape(-1, 2)  # (H*W, 2)
            ref_points_list.append(ref)
        ref_points = torch.cat(ref_points_list, dim=0)  # (sum(HW), 2)
        ref_points = ref_points.unsqueeze(0).unsqueeze(2).repeat(B, 1, self.n_levels, 1)  # (B, sum(HW), n_levels, 2)

        memory = src_flat
        for layer in self.encoder_layers:
            memory = layer(memory, ref_points, spatial_shapes, pos_flat)

        # decoder
        query_pos = self.query_pos_embed.weight.unsqueeze(0).repeat(B, 1, 1)   # (B, Q, C)
        tgt = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)             # (B, Q, C)
        dec_ref_points = self.reference_points_head(query_pos).sigmoid()       # (B, Q, 2)
        dec_ref_points = dec_ref_points.unsqueeze(2).repeat(1, 1, self.n_levels, 1)  # (B, Q, n_levels, 2)

        all_class_logits, all_bbox_preds = [], []
        for layer in self.decoder_layers:
            tgt = layer(tgt, query_pos, dec_ref_points, memory, spatial_shapes)
            all_class_logits.append(self.class_head(tgt))
            all_bbox_preds.append(self.bbox_head(tgt).sigmoid())  # normalize [0,1]: cx,cy,w,h

        return {
            "pred_logits": all_class_logits[-1],   # (B, Q, num_classes)  -- output cuối dùng để inference
            "pred_boxes": all_bbox_preds[-1],      # (B, Q, 4)
            "aux_logits": all_class_logits[:-1],   # deep supervision cho các layer trước
            "aux_boxes": all_bbox_preds[:-1],
        }


# ============================================================
# 5. HUNGARIAN MATCHER + LOSS (có weighted class cho imbalance)
# ============================================================

def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_box_iou(boxes1, boxes2):
    boxes1_xyxy = box_cxcywh_to_xyxy(boxes1)
    boxes2_xyxy = box_cxcywh_to_xyxy(boxes2)

    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]).clamp(0) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1]).clamp(0)
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]).clamp(0) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1]).clamp(0)

    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    lt_c = torch.min(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    rb_c = torch.max(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1]

    return iou - (area_c - union) / area_c.clamp(min=1e-6)


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, pred_logits, pred_boxes, targets):
        B, Q, num_classes = pred_logits.shape
        indices = []
        for b in range(B):
            tgt_labels = targets[b]["labels"]
            tgt_boxes = targets[b]["boxes"]
            if len(tgt_labels) == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue

            out_prob = pred_logits[b].sigmoid()  # (Q, num_classes)
            out_bbox = pred_boxes[b]              # (Q, 4)

            cost_class = -out_prob[:, tgt_labels]  # (Q, num_tgt)
            cost_bbox = torch.cdist(out_bbox, tgt_boxes, p=1)  # (Q, num_tgt)
            cost_giou = -generalized_box_iou(out_bbox, tgt_boxes)

            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.cpu()
            row_ind, col_ind = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)))
        return indices


class SetCriterion(nn.Module):
    """Loss: Focal loss (weighted per-class cho imbalance) + L1 bbox + GIoU."""

    def __init__(self, num_classes, class_weights=None, matcher=None,
                 weight_class=2.0, weight_bbox=5.0, weight_giou=2.0, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher or HungarianMatcher()
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        # class_weights: tensor (num_classes,) — trọng số nhân thêm cho class hiếm (glass, metal...)
        self.register_buffer("class_weights", class_weights if class_weights is not None
                              else torch.ones(num_classes))

    def focal_loss(self, pred_logits, targets_onehot, class_weight_per_pos):
        prob = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, targets_onehot, reduction="none")
        p_t = prob * targets_onehot + (1 - prob) * (1 - targets_onehot)
        loss = ce * ((1 - p_t) ** self.focal_gamma)
        alpha_t = self.focal_alpha * targets_onehot + (1 - self.focal_alpha) * (1 - targets_onehot)
        loss = alpha_t * loss
        loss = loss * class_weight_per_pos.unsqueeze(0).unsqueeze(0)  # nhân weight imbalance theo class
        return loss.sum()

    def forward(self, outputs, targets):
        indices = self.matcher(outputs["pred_logits"], outputs["pred_boxes"], targets)
        B, Q, C = outputs["pred_logits"].shape

        targets_onehot = torch.zeros(B, Q, C, device=outputs["pred_logits"].device)
        loss_bbox, loss_giou = 0.0, 0.0
        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)

        for b, (row_ind, col_ind) in enumerate(indices):
            if len(row_ind) == 0:
                continue
            labels = targets[b]["labels"][col_ind]
            targets_onehot[b, row_ind, labels] = 1.0

            pred_b = outputs["pred_boxes"][b, row_ind]
            tgt_b = targets[b]["boxes"][col_ind]
            loss_bbox += F.l1_loss(pred_b, tgt_b, reduction="sum")
            loss_giou += (1 - torch.diag(generalized_box_iou(pred_b, tgt_b))).sum()

        loss_class = self.focal_loss(outputs["pred_logits"], targets_onehot, self.class_weights) / num_boxes
        loss_bbox = loss_bbox / num_boxes
        loss_giou = loss_giou / num_boxes

        total = self.weight_class * loss_class + self.weight_bbox * loss_bbox + self.weight_giou * loss_giou

        # deep supervision: cộng thêm loss ở các decoder layer trung gian
        for aux_logits, aux_boxes in zip(outputs.get("aux_logits", []), outputs.get("aux_boxes", [])):
            aux_indices = self.matcher(aux_logits, aux_boxes, targets)
            aux_onehot = torch.zeros(B, Q, C, device=aux_logits.device)
            aux_lb, aux_lg = 0.0, 0.0
            for b, (row_ind, col_ind) in enumerate(aux_indices):
                if len(row_ind) == 0:
                    continue
                labels = targets[b]["labels"][col_ind]
                aux_onehot[b, row_ind, labels] = 1.0
                pred_b = aux_boxes[b, row_ind]
                tgt_b = targets[b]["boxes"][col_ind]
                aux_lb += F.l1_loss(pred_b, tgt_b, reduction="sum")
                aux_lg += (1 - torch.diag(generalized_box_iou(pred_b, tgt_b))).sum()
            aux_lc = self.focal_loss(aux_logits, aux_onehot, self.class_weights) / num_boxes
            total += self.weight_class * aux_lc + self.weight_bbox * (aux_lb / num_boxes) + self.weight_giou * (aux_lg / num_boxes)

        return total, {"loss_class": loss_class.item(), "loss_bbox": loss_bbox.item(), "loss_giou": loss_giou.item()}


# ============================================================
# 6. DATASET (đọc annotations_coco.json vừa convert)
# ============================================================

class CocoWasteDataset(Dataset):
    def __init__(self, coco_json: str, images_dir: str, img_size: int = 518):
        with open(coco_json, "r") as f:
            coco = json.load(f)

        self.images_dir = images_dir
        self.img_size = img_size
        self.images = {im["id"]: im for im in coco["images"]}
        self.img_ids = list(self.images.keys())

        self.anns_by_image = {}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # chuẩn DINOv2/ImageNet
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_info["width"], img_info["height"]

        anns = self.anns_by_image.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            boxes.append([cx, cy, nw, nh])
            labels.append(ann["category_id"])

        image = self.transform(image)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
        }
        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


# ============================================================
# 7. TRAINING LOOP
# ============================================================

def compute_class_weights(coco_json_path: str, num_classes: int) -> torch.Tensor:
    with open(coco_json_path, "r") as f:
        coco = json.load(f)
    counts = [0] * num_classes
    for ann in coco["annotations"]:
        counts[ann["category_id"]] += 1
    total = sum(counts)
    weights = [total / (num_classes * max(c, 1)) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_json", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--val_coco_json", default=None)
    parser.add_argument("--val_images_dir", default=None)
    parser.add_argument("--img_size", type=int, default=518)  # phải chia hết cho 14
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--out_dir", default="checkpoints")
    args = parser.parse_args()

    assert args.img_size % 14 == 0, "img_size phải chia hết cho 14 (patch size của DINOv2)"
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dùng thiết bị: {device}")

    train_ds = CocoWasteDataset(args.coco_json, args.images_dir, args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=2)

    class_weights = compute_class_weights(args.coco_json, args.num_classes).to(device)
    print("Class weights (imbalance handling):", class_weights.tolist())

    model = DINOv2DeformableDETR(
        num_classes=args.num_classes, num_queries=args.num_queries, freeze_backbone=args.freeze_backbone
    ).to(device)

    criterion = SetCriterion(args.num_classes, class_weights=class_weights).to(device)

    # chỉ optimize phần head (backbone frozen) — lọc theo requires_grad
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]

            outputs = model(imgs)
            loss, loss_dict = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | {loss_dict}")

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(args.out_dir, f"model_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Đã lưu checkpoint: {ckpt_path}")

    print("Training hoàn tất.")


if __name__ == "__main__":
    main()
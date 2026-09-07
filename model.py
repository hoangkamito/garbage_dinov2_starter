import torch
import torch.nn as nn

class DINOv2Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_name: str = "dinov2_vits14",
        mode: str = "linear",
        unfreeze_blocks: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.model_name = model_name
        self.mode = mode
        self.unfreeze_blocks = unfreeze_blocks

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=True,
        )

        embed_dim = self.backbone.embed_dim

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self.configure_trainable(mode, unfreeze_blocks)

    def configure_trainable(self, mode: str, unfreeze_blocks: int = 2):
        # Freeze toàn bộ backbone trước.
        for p in self.backbone.parameters():
            p.requires_grad = False

        if mode == "linear":
            return

        if mode != "finetune":
            raise ValueError("mode phải là 'linear' hoặc 'finetune'.")

        # Fine-tune một số transformer blocks cuối.
        n_blocks = len(self.backbone.blocks)
        n = min(max(unfreeze_blocks, 1), n_blocks)

        for block in self.backbone.blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

        # Cho phép LayerNorm cuối thích nghi.
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

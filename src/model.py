import torch
import torch.nn as nn

from transformers import AutoModel


class DINOv2Classifier(nn.Module):
    def __init__(
        self,
        num_classes=10,
        model_name="facebook/dinov2-base",
        dropout=0.2
    ):
        super().__init__()

        # DINOv2 pretrained backbone
        self.backbone = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = self.backbone.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

        self.backbone_frozen = False

    def freeze_backbone(self):
        """
        Freeze toàn bộ DINOv2 backbone.
        Dùng khi inference hoặc linear probing.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone_frozen = True

    def unfreeze_backbone(self):
        """
        Unfreeze toàn bộ backbone.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.backbone_frozen = False

    def forward(self, x):
        # Nếu backbone bị freeze thì không cần gradient
        if self.backbone_frozen:

            self.backbone.eval()

            with torch.no_grad():
                outputs = self.backbone(
                    pixel_values=x
                )

        else:
            outputs = self.backbone(
                pixel_values=x
            )

        # CLS token
        features = outputs.last_hidden_state[:, 0, :]

        # Classification head
        logits = self.classifier(features)

        return logits
"""
Model Architecture for Ghost in the Machine
=============================================
Two model classes:

1. ``SiamesePromptDetector`` – Siamese network with shared BERT encoder
   for semantic anomaly detection via contrastive learning.
2. ``BaselineClassifier`` – vanilla BERT fine-tuned classifier used as
   a comparison baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class SiamesePromptDetector(nn.Module):
    """Siamese network for prompt injection detection.

    Architecture:
        1. Shared BERT encoder (frozen first N layers optional)
        2. Mean pooling over token embeddings
        3. Projection head:  768 -> 512 -> ReLU -> Dropout -> 256
        4. Contrastive loss operates on the projected embeddings

    At inference time, a single text is embedded and classified by
    comparing its distance to learned class centroids.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 256,
        dropout: float = 0.1,
        freeze_layers: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size: int = self.encoder.config.hidden_size  # typically 768

        # Optionally freeze early transformer layers
        if freeze_layers > 0:
            for i, layer in enumerate(self.encoder.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )

        # Classification head (binary) operating on the projected embedding
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    # ------------------------------------------------------------------
    # Core forward helpers
    # ------------------------------------------------------------------
    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a single text through BERT + projection head.

        Returns:
            Embedding of shape ``(batch, embedding_dim)``.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling (mask-aware)
        token_embeddings = outputs.last_hidden_state  # (B, seq, hidden)
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, seq, 1)
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_embeddings / counts  # (B, hidden)
        return self.projection(pooled)  # (B, embedding_dim)

    # ------------------------------------------------------------------
    # Siamese forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids_1: torch.Tensor,
        attention_mask_1: torch.Tensor,
        input_ids_2: torch.Tensor,
        attention_mask_2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a pair of texts.

        Returns:
            ``(embedding_1, embedding_2)`` each of shape ``(batch, embedding_dim)``.
        """
        emb1 = self._encode(input_ids_1, attention_mask_1)
        emb2 = self._encode(input_ids_2, attention_mask_2)
        return emb1, emb2

    # ------------------------------------------------------------------
    # Single-text classification
    # ------------------------------------------------------------------
    def classify(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Classify a single text. Returns logits of shape ``(batch, 2)``."""
        emb = self._encode(input_ids, attention_mask)
        return self.classifier(emb)


# ======================================================================
# Contrastive Loss
# ======================================================================
class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese training.

    L = (1 - Y) * 0.5 * D^2  +  Y * 0.5 * max(0, margin - D)^2

    where Y=1 means *same* class, Y=0 means *different* class,
    and D is the Euclidean distance between the two embeddings.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        pair_label: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            emb1, emb2: Embeddings of shape ``(batch, dim)``.
            pair_label: 1.0 if same class, 0.0 if different class.
        """
        dist = F.pairwise_distance(emb1, emb2)  # (batch,)
        loss_same = pair_label * 0.5 * dist.pow(2)
        loss_diff = (1 - pair_label) * 0.5 * F.relu(self.margin - dist).pow(2)
        return (loss_same + loss_diff).mean()


# ======================================================================
# Baseline Classifier
# ======================================================================
class BaselineClassifier(nn.Module):
    """Simple BERT fine-tuned binary classifier for comparison.

    Uses CLS token -> Linear(hidden, 2).
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size: int = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass. Returns logits of shape ``(batch, 2)``."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # CLS
        return self.classifier(cls_token)

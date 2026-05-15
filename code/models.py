"""Teacher and Student models for fee anomaly analysis/distillation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelBatch:
    num_x: torch.Tensor
    cat_x: torch.Tensor
    reason_x: torch.Tensor
    llm_x: torch.Tensor | None = None
    # Phase-6: optional explicit environment view for the student. The teacher
    # ignores `env_x` (it already aggregates env features into its num view),
    # but the student routes them through a dedicated environment encoder.
    env_x: Optional[torch.Tensor] = None


class CategoricalEncoder(nn.Module):
    def __init__(self, cardinalities: List[int], emb_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(max(2, card), emb_dim) for card in cardinalities]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C]
        outputs = []
        for idx, emb in enumerate(self.embeddings):
            outputs.append(emb(x[:, idx]))
        if not outputs:
            return x.new_zeros((x.size(0), 0), dtype=torch.float32)
        return torch.cat(outputs, dim=1)


class TeacherModel(nn.Module):
    """Rule-enhanced multi-view teacher model.

    Views:
    - Numeric view (monthly features + lag features)
    - Categorical view (customer/profile/tariff categories)
    - Reason view (rule-hit multi-hot)
    """

    def __init__(
        self,
        num_dim: int,
        cat_cardinalities: List[int],
        reason_dim: int,
        llm_dim: int = 0,
        reason_label_dim: int | None = None,
        hidden_dim: int = 128,
        cat_emb_dim: int = 16,
    ) -> None:
        super().__init__()
        self.cat_encoder = CategoricalEncoder(cat_cardinalities, emb_dim=cat_emb_dim)
        cat_out_dim = len(cat_cardinalities) * cat_emb_dim

        self.num_proj = nn.Sequential(
            nn.Linear(num_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.cat_proj = nn.Sequential(
            nn.Linear(cat_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.reason_proj = nn.Sequential(
            nn.Linear(reason_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.use_llm_view = llm_dim > 0
        if self.use_llm_view:
            self.llm_proj = nn.Sequential(
                nn.Linear(llm_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        self.n_views = 4 if self.use_llm_view else 3

        # Gating fusion between views.
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * self.n_views, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.n_views),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.anomaly_head = nn.Linear(hidden_dim, 1)
        # Reason supervision dimension can be different from reason input view dimension.
        self.reason_head = nn.Linear(hidden_dim, reason_label_dim if reason_label_dim is not None else reason_dim)

    def forward(self, batch: ModelBatch) -> Dict[str, torch.Tensor]:
        z_num = self.num_proj(batch.num_x)
        z_cat = self.cat_proj(self.cat_encoder(batch.cat_x))
        z_reason = self.reason_proj(batch.reason_x)
        views = [z_num, z_cat, z_reason]
        if self.use_llm_view:
            if batch.llm_x is None:
                raise ValueError("TeacherModel configured with llm_dim > 0 but batch.llm_x is None.")
            views.append(self.llm_proj(batch.llm_x))

        gates = F.softmax(self.gate(torch.cat(views, dim=1)), dim=1)
        z = sum(gates[:, idx : idx + 1] * view for idx, view in enumerate(views))
        z = self.fusion(z)

        anomaly_logit = self.anomaly_head(z).squeeze(1)
        reason_logit = self.reason_head(z)
        return {
            "repr": z,
            "anomaly_logit": anomaly_logit,
            "reason_logit": reason_logit,
            "gate": gates,
        }


class SparseGate(nn.Module):
    """Feature-wise sparse gate for tiny student."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.logits).unsqueeze(0)
        return x * gate

    def l1_regularization(self) -> torch.Tensor:
        return torch.sigmoid(self.logits).mean()


class EnvironmentEncoder(nn.Module):
    """Phase-6 dedicated environment encoder for the student.

    Routes `env_self_*` / `env_peer_*` / `env_tariff_*` / `env_season_*` columns
    through a small MLP separate from the main numeric trunk so the student
    explicitly conditions its latent on dynamic environment context (per
    `phase.md` Phase-6 spec, mirroring the E2P "user environment" view).
    """

    def __init__(self, env_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.env_dim = int(env_dim)
        if self.env_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(self.env_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            self.out_dim = hidden_dim
        else:
            self.proj = None
            self.out_dim = 0

    def forward(self, env_x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.proj is None or env_x is None or env_x.size(1) == 0:
            return None
        return self.proj(env_x)


class StudentModel(nn.Module):
    """Tiny student model for low-latency deployment.

    Phase-6 upgrades vs. earlier versions:
    - Optional `env_dim > 0` enables an EnvironmentEncoder branch fused with
      the main numeric trunk before the latent head.
    - When `latent_dim > 0`, an `(mu_head, logvar_head)` pair produces a
      reparameterized `z`; anomaly/reason heads consume `z` instead of the
      raw encoder output, enabling KL regularization.
    - When `prefix_dim > 0`, a `prefix_proj(z)` head is exposed so the
      training loop can align the student latent with the LLM teacher's
      `llm_prefix_emb_*` (E2P-style prefix distillation).
    The model stays backwards-compatible: passing `env_dim=0`, `latent_dim=0`
    and `prefix_dim=0` reproduces the pre-phase-6 behaviour byte-for-byte.
    """

    def __init__(
        self,
        num_dim: int,
        hidden_dim: int = 64,
        reason_dim: int = 2,
        teacher_repr_dim: int = 128,
        env_dim: int = 0,
        latent_dim: int = 0,
        prefix_dim: int = 0,
    ) -> None:
        super().__init__()
        self.sparse_gate = SparseGate(num_dim)
        self.encoder = nn.Sequential(
            nn.Linear(num_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.env_encoder = EnvironmentEncoder(env_dim=env_dim, hidden_dim=hidden_dim)
        fused_in = hidden_dim + self.env_encoder.out_dim
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.latent_dim = int(latent_dim)
        if self.latent_dim > 0:
            self.mu_head = nn.Linear(hidden_dim, self.latent_dim)
            self.logvar_head = nn.Linear(hidden_dim, self.latent_dim)
            head_in_dim = self.latent_dim
        else:
            self.mu_head = None
            self.logvar_head = None
            head_in_dim = hidden_dim

        self.repr_proj = nn.Linear(head_in_dim, teacher_repr_dim)
        self.anomaly_head = nn.Linear(head_in_dim, 1)
        self.reason_head = nn.Linear(head_in_dim, reason_dim)

        self.prefix_dim = int(prefix_dim)
        if self.prefix_dim > 0:
            self.prefix_proj = nn.Linear(head_in_dim, self.prefix_dim)
        else:
            self.prefix_proj = None

    def forward(
        self,
        num_x: torch.Tensor,
        env_x: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        gated = self.sparse_gate(num_x)
        h_num = self.encoder(gated)
        h_env = self.env_encoder(env_x)
        if h_env is not None:
            h = torch.cat([h_num, h_env], dim=1)
        else:
            h = h_num
        h = self.fuse(h)

        if self.mu_head is not None and self.logvar_head is not None:
            mu = self.mu_head(h)
            logvar = self.logvar_head(h).clamp(min=-8.0, max=8.0)
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + std * eps
            else:
                z = mu
        else:
            mu = h
            logvar = None
            z = h

        anomaly_logit = self.anomaly_head(z).squeeze(1)
        reason_logit = self.reason_head(z)
        z_proj = self.repr_proj(z)

        out: Dict[str, torch.Tensor] = {
            "repr": z,
            "mu": mu,
            "repr_for_distill": z_proj,
            "anomaly_logit": anomaly_logit,
            "reason_logit": reason_logit,
        }
        if logvar is not None:
            out["logvar"] = logvar
        if self.prefix_proj is not None:
            out["prefix_pred"] = self.prefix_proj(z)
        return out

    def kl_regularization(
        self,
        mu: torch.Tensor,
        logvar: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """KL(N(mu, sigma^2) || N(0, I)) averaged over batch.

        Returns a 0-tensor when the student is configured without a latent
        head (logvar is None), which keeps the training loop branch-free.
        """
        if logvar is None or self.latent_dim <= 0:
            return mu.new_zeros(())
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        return kl.mean()


def prefix_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "mse",
) -> torch.Tensor:
    """Mean MSE / 1-cosine between student prefix prediction and LLM prefix.

    `mask` is a 1D float tensor (B,) marking rows that actually carry an LLM
    prefix (real OpenAI rows). Rows without LLM coverage contribute 0 and do
    not pollute the gradient. Returns 0 when the batch has no covered row.
    """
    if pred.numel() == 0 or target.numel() == 0:
        return pred.new_zeros(())
    mask = mask.to(pred.dtype)
    denom = mask.sum().clamp(min=1.0)
    if mode == "cosine":
        cos = F.cosine_similarity(pred, target, dim=1).clamp(min=-1.0, max=1.0)
        per_row = (1.0 - cos)
    else:
        per_row = ((pred - target) ** 2).mean(dim=1)
    return (per_row * mask).sum() / denom


def distill_loss(
    student_out: Dict[str, torch.Tensor],
    teacher_out: Dict[str, torch.Tensor],
    y_weak: torch.Tensor,
    y_reason: torch.Tensor,
    *,
    llm_prefix_target: Optional[torch.Tensor] = None,
    llm_prefix_mask: Optional[torch.Tensor] = None,
    alpha_prob: float = 1.0,
    alpha_reason: float = 1.0,
    alpha_repr: float = 0.5,
    alpha_sup: float = 1.0,
    alpha_kl: float = 0.0,
    alpha_prefix: float = 0.0,
    prefix_mode: str = "mse",
) -> Dict[str, torch.Tensor]:
    l_sup = F.binary_cross_entropy_with_logits(student_out["anomaly_logit"], y_weak)

    t_prob = torch.sigmoid(teacher_out["anomaly_logit"]).detach()
    s_prob = torch.sigmoid(student_out["anomaly_logit"])
    l_prob = F.mse_loss(s_prob, t_prob)

    t_reason = torch.sigmoid(teacher_out["reason_logit"]).detach()
    s_reason = torch.sigmoid(student_out["reason_logit"])
    l_reason_distill = F.mse_loss(s_reason, t_reason)
    l_reason_sup = F.binary_cross_entropy_with_logits(student_out["reason_logit"], y_reason)
    l_reason = l_reason_distill + l_reason_sup

    l_repr = F.mse_loss(student_out["repr_for_distill"], teacher_out["repr"].detach())

    mu = student_out.get("mu")
    logvar = student_out.get("logvar")
    if alpha_kl > 0.0 and mu is not None and logvar is not None:
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    else:
        kl = (mu if mu is not None else student_out["anomaly_logit"]).new_zeros(())

    if (
        alpha_prefix > 0.0
        and llm_prefix_target is not None
        and llm_prefix_mask is not None
        and "prefix_pred" in student_out
    ):
        l_prefix = prefix_alignment_loss(
            student_out["prefix_pred"],
            llm_prefix_target.detach(),
            llm_prefix_mask,
            mode=prefix_mode,
        )
    else:
        l_prefix = student_out["anomaly_logit"].new_zeros(())

    total = (
        alpha_sup * l_sup
        + alpha_prob * l_prob
        + alpha_reason * l_reason
        + alpha_repr * l_repr
        + alpha_kl * kl
        + alpha_prefix * l_prefix
    )
    return {
        "total": total,
        "l_sup": l_sup,
        "l_prob": l_prob,
        "l_reason": l_reason,
        "l_repr": l_repr,
        "l_kl": kl,
        "l_prefix": l_prefix,
    }

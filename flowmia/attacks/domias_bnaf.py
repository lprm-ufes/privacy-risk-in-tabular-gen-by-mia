"""
DOMIAS (Density-based Overfitting Membership Inference Attack on Synthetic data).

Uses a Block Neural Autoregressive Flow (BNAF) to estimate the log-density of
the synthetic distribution. Points with higher log-density are predicted to be
training members. An optional reference model (fitted on non-member data) can
be used to compute a density ratio, which corrects for distributional shift.

Reference:
    van Breugel et al., "Membership Inference Attacks Against Synthetic Data",
    NeurIPS 2023.
"""

import gc
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from pathlib import Path
from flowmia.attacks.domias.domias import density_estimator_trainer, compute_log_p_x


def _predict_log_p_batched(
    model: torch.nn.Module,
    data: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """Compute log p(x) in mini-batches under torch.no_grad() to prevent CUDA OOM."""
    model.eval()
    log_probs = []
    eval_batch_size = max(1, min(batch_size, 128) if batch_size else 128)
    with torch.no_grad():
        for i in range(0, len(data), eval_batch_size):
            batch = torch.tensor(data[i : i + eval_batch_size], dtype=torch.float32, device=device)
            log_p = compute_log_p_x(model, batch).detach().cpu()
            log_probs.append(log_p)
    return torch.cat(log_probs).numpy()


def domias_bnaf(
    train_members: np.ndarray,
    non_members: np.ndarray,
    synthetic_data: np.ndarray,
    device: torch.device,
    save_path: str = "",
    epochs: int = 10,
    batch_size: int = 50,
    reference_data: np.ndarray = None,
    load=False
) -> tuple[np.ndarray, float]:
    """
    Compute DOMIAS membership inference scores using a normalizing flow density model.

    A BNAF model is trained on ``synthetic_data`` to approximate its density
    p_S. The log-density log p_S(x) is then evaluated at every query point.
    If ``reference_data`` is provided, a second model is trained on it to
    estimate p_R, and the final score becomes log p_S(x) − log p_R(x) (the
    DOMIAS formulation). Without reference data, log p_S alone is used.

    Args:
        train_members: Pre-transformed member samples, shape (N, D).
        non_members: Pre-transformed non-member samples, shape (M, D).
        synthetic_data: Pre-transformed synthetic samples used to fit the
            density model, shape (S, D).
        device: Torch device for model inference.
        reference_data: Optional pre-transformed reference (non-member)
            samples used to fit a calibration density model p_R. When
            provided, the score is the log density ratio log p_S / p_R.

    Returns:
        Tuple of:
            - scores (np.ndarray): Attack scores for members followed by
              non-members, shape (N + M,).
            - auc (float): ROC-AUC of the attack.
    """
    X_test = np.vstack([train_members, non_members])
    y_test = np.concatenate([np.ones(len(train_members)), np.zeros(len(non_members))])

    # Fit density model on synthetic data (train/val split at midpoint)
    mid = len(synthetic_data) // 2
    p_S_func, p_S_model = density_estimator_trainer(
        synthetic_data,
        synthetic_data[:mid],
        synthetic_data[mid:],
        batch_dim=batch_size,
        epochs=epochs,
        load=load,
        workspace=Path(save_path)
    )
    del p_S_func
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log_p_S = _predict_log_p_batched(p_S_model, X_test, device=device, batch_size=batch_size)

    # Free synthetic model from GPU before optionally training reference model
    del p_S_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if reference_data is not None:
        # Density ratio variant: corrects for distributional shift
        mid_r = len(reference_data) // 2
        p_R_func, p_R_model = density_estimator_trainer(
            reference_data,
            reference_data[:mid_r],
            reference_data[mid_r:],
            batch_dim=batch_size,
            epochs=epochs,
            load=load,
            workspace=Path(save_path)
        )
        del p_R_func
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log_p_R = _predict_log_p_batched(p_R_model, X_test, device=device, batch_size=batch_size)

        del p_R_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        scores = log_p_S - log_p_R
    else:
        scores = log_p_S

    auc = roc_auc_score(y_test, scores)
    return scores, auc


# ------------------------------------------------------------------
# BaseAttack adapter
# ------------------------------------------------------------------

from flowmia.attacks.base import AttackResult, BaseAttack
from flowmia.attacks.registry import register_attack


@register_attack("domias")
class DomiasAttack(BaseAttack):
    """BaseAttack adapter around the domias_bnaf() function."""

    def __init__(self, device=None, **hyperparams):
        super().__init__(device=device, **hyperparams)
        self._X_synth: np.ndarray = None
        self._reference_data: np.ndarray = None

    def fit(
        self,
        X_member: np.ndarray,
        X_non_member: np.ndarray,
        X_synth: np.ndarray,
        reference_data: np.ndarray = None,
        **_,
    ) -> "DomiasAttack":
        self._X_synth = X_synth
        self._reference_data = reference_data
        return self

    def attack(
        self,
        X_member: np.ndarray,
        X_non_member: np.ndarray,
        test_size: int = 1000,
        epochs: int = 10,
        batch_size: int = 50,
        save_path: str = "",
        load: bool = False,
        seed: int = 42,
        **kwargs,
    ) -> AttackResult:
        rng = np.random.default_rng(seed)
        idx_m = rng.choice(len(X_member), size=test_size, replace=False)
        idx_nm = rng.choice(len(X_non_member), size=test_size, replace=False)
        scores, auc = domias_bnaf(
            train_members=X_member[idx_m],
            non_members=X_non_member[idx_nm],
            synthetic_data=self._X_synth,
            device=self.device,
            save_path=save_path,
            epochs=epochs,
            batch_size=batch_size,
            reference_data=self._reference_data,
            load=load,
        )
        return AttackResult(scores=scores, auc=auc)
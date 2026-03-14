"""
Chronos-2 foundation model pipeline for LSST classification.

M7: Zero-shot embedding extraction + sklearn classifier
M8: Pre-computed features + trainable MLP head (encoder frozen)
M9: LP-FT with LoRA — Linear Probe then Fine-Tune with Low-Rank Adaptation

Chronos-2 (Amazon, 2025) is a patch-based forecasting FM with native multivariate
support via alternating time-attention and group-attention layers.

Key advantages over MOMENT/TiReX for LSST:
  - Native multivariate: cross-channel attention captures band correlations
  - No RevIN: tokenizes via quantile binning, amplitude info preserved
  - Flexible length: no forced 512 padding — 36 timesteps → 5 tokens naturally

Architecture: d_model=768, 12 layers, patch_size=16
36 timesteps × 6 channels → (6, 5 tokens, 768) → pooled to (6 × 768) = 4608-dim
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from chronos import Chronos2Pipeline
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score

SEED = 42
CHRONOS_D_MODEL = 768


# ──────────────────────────────────────────────
# M7: Zero-shot embedding extraction
# ──────────────────────────────────────────────

def extract_chronos_embeddings(X_cf, model_name="amazon/chronos-2",
                                batch_size=64, device="cuda",
                                pool_variates="flatten"):
    """Extract Chronos-2 embeddings for multivariate classification.

    Chronos-2 natively handles multivariate input: (n_variates, T).
    Cross-variate attention captures inter-channel relationships.

    Args:
        X_cf: (n, C, T) channels-first numpy array
        model_name: HuggingFace model path
        batch_size: samples per forward pass
        device: compute device
        pool_variates: how to combine variate embeddings
            "flatten": concatenate → (n, C * d_model)
            "mean": average across variates → (n, d_model)

    Returns:
        numpy array of shape (n, embedding_dim)
    """
    if isinstance(X_cf, torch.Tensor):
        X_cf = X_cf.cpu().numpy()

    n, C, T = X_cf.shape
    X_cf = X_cf.astype(np.float32)

    pipeline = Chronos2Pipeline.from_pretrained(model_name, device_map=device)

    all_embeddings = []
    for i in range(0, n, batch_size):
        batch = [X_cf[j] for j in range(i, min(i + batch_size, n))]  # list of (C, T)
        emb_list, _ = pipeline.embed(batch)

        batch_embs = []
        for emb in emb_list:
            # emb: (C, num_tokens, d_model)
            # Mean-pool over tokens → (C, d_model)
            pooled_tokens = emb.mean(dim=1)

            if pool_variates == "flatten":
                # Concatenate channels → (C * d_model,)
                batch_embs.append(pooled_tokens.reshape(-1).cpu().numpy())
            elif pool_variates == "mean":
                # Average channels → (d_model,)
                batch_embs.append(pooled_tokens.mean(dim=0).cpu().numpy())
            else:
                raise ValueError(f"Unknown pool_variates: {pool_variates}")

        all_embeddings.append(np.stack(batch_embs))

        if (i // batch_size) % 5 == 0:
            print(f"  Embedded {min(i + batch_size, n)}/{n} samples")

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Chronos-2 embeddings: {embeddings.shape}")
    return embeddings


# ──────────────────────────────────────────────
# M9: LP-FT with LoRA
# ──────────────────────────────────────────────

class Chronos2Classifier(nn.Module):
    """Chronos-2 backbone + classification head for end-to-end fine-tuning.

    Wraps the Chronos2Model, runs encode(), pools hidden states,
    and feeds into a linear classification head.

    For multivariate input: each sample's 6 channels are passed as
    independent series in the batch with shared group_ids, so
    cross-variate attention mixes information across bands.
    """

    def __init__(self, backbone, n_classes, n_channels=6, d_model=768):
        super().__init__()
        self.backbone = backbone
        self.n_channels = n_channels
        self.d_model = d_model
        # Linear head (no hidden layer — regularization via LoRA + weight decay)
        self.head = nn.Linear(n_channels * d_model, n_classes)

    def _encode(self, x_cf):
        """x_cf: (batch, n_channels, T) float tensor on device.
        Returns: (batch, n_channels * d_model) pooled embeddings.
        """
        bs, C, T = x_cf.shape
        # Flatten channels into batch dim: (bs * C, T)
        x_flat = x_cf.reshape(bs * C, T)

        # Group IDs: channels of the same sample share a group
        # e.g., for bs=2, C=6: [0,0,0,0,0,0, 1,1,1,1,1,1]
        group_ids = torch.arange(bs, device=x_flat.device).repeat_interleave(C)

        # Run encoder (gradients flow through LoRA adapters)
        encoder_outputs, loc_scale, _, num_ctx_patches = self.backbone.encode(
            context=x_flat,
            group_ids=group_ids,
            num_output_patches=0,
        )
        hidden = encoder_outputs[0]  # (bs*C, num_tokens, d_model)

        # Mean-pool over tokens → (bs*C, d_model)
        pooled = hidden.mean(dim=1)

        # Reshape back: (bs, C, d_model) → (bs, C * d_model)
        return pooled.reshape(bs, C * self.d_model)

    def forward(self, x_cf):
        return self.head(self._encode(x_cf))


def load_chronos_for_lora(n_channels=6, n_classes=14, lora_rank=4,
                           model_name="amazon/chronos-2", device="cuda"):
    """Load Chronos-2, inject LoRA adapters into attention q/v, add classification head.

    Returns the Chronos2Classifier with LoRA-adapted backbone.
    All backbone params frozen except LoRA adapters.
    """
    pipeline = Chronos2Pipeline.from_pretrained(model_name, device_map=device)
    backbone = pipeline.model

    # Target modules: q and v projections in both time-attention and group-attention
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()

    model = Chronos2Classifier(backbone, n_classes, n_channels)
    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    return model


def finetune_chronos_lpft(model, X_train, y_train, X_val, y_val,
                           class_weights, n_classes=14,
                           lp_epochs=30, ft_epochs=70,
                           lp_lr=1e-3, ft_lr=1e-4,
                           batch_size=16, patience=15,
                           weight_decay=1e-2, device="cuda"):
    """LP-FT: Linear Probing then Fine-Tuning with LoRA.

    Stage 1 (LP): Freeze backbone (including LoRA). Train only the head.
                  Stabilizes the head weights before gradients flow into backbone.
    Stage 2 (FT): Unfreeze LoRA adapters. Fine-tune LoRA + head jointly
                  with a lower learning rate.

    Args:
        model: Chronos2Classifier with LoRA backbone
        X_train/X_val: (n, C, T) numpy arrays
        y_train/y_val: integer labels
        class_weights: numpy array of class weights
        lp_epochs/ft_epochs: epochs for each stage
        lp_lr/ft_lr: learning rates for each stage
    """
    cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(Xt, yt)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": [], "val_wf1": [], "stage": []}
    best_wf1 = 0.0
    best_state = None
    wait = 0

    def _run_stage(stage_name, epochs, lr, train_lora):
        nonlocal best_wf1, best_state, wait

        # Freeze/unfreeze LoRA params
        for name, param in model.backbone.named_parameters():
            if "lora_" in name:
                param.requires_grad = train_lora
            else:
                param.requires_grad = False

        # Head is always trainable
        for param in model.head.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'='*50}")
        print(f"Stage: {stage_name} | LR: {lr} | Trainable: {trainable:,}")
        print(f"{'='*50}")

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            model.train()
            epoch_loss, n_batches = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            history["train_loss"].append(epoch_loss / n_batches)
            history["stage"].append(stage_name)

            # Validation
            model.eval()
            with torch.no_grad():
                all_logits = []
                for i in range(0, len(Xv), batch_size):
                    xb = Xv[i:i + batch_size].to(device)
                    all_logits.append(model(xb).cpu())
                logits_v = torch.cat(all_logits, dim=0)
                val_loss = criterion(logits_v.to(device), yv.to(device)).item()
                y_pred_v = logits_v.argmax(1).numpy()
                wf1 = f1_score(y_val, y_pred_v, average="weighted")

            history["val_loss"].append(val_loss)
            history["val_wf1"].append(wf1)

            if wf1 > best_wf1:
                best_wf1 = wf1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 5 == 0:
                print(f"  [{stage_name}] Epoch {epoch+1}/{epochs} — "
                      f"train_loss={history['train_loss'][-1]:.4f}  "
                      f"val_loss={val_loss:.4f}  val_wf1={wf1:.4f}")

    # ── Stage 1: Linear Probing (head only) ──
    _run_stage("LP", lp_epochs, lp_lr, train_lora=False)

    # ── Stage 2: Fine-Tuning (LoRA + head) ──
    wait = 0  # reset patience for stage 2
    _run_stage("FT", ft_epochs, ft_lr, train_lora=True)

    # Restore best weights
    model.load_state_dict(best_state)
    model.to(device)
    print(f"\nBest val W-F1: {best_wf1:.4f}")
    return model, history


def predict_chronos(model, X, batch_size=16, device="cuda"):
    """Inference with Chronos2Classifier. X: (n, C, T) numpy array."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    all_preds = []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i:i + batch_size].to(device)
        with torch.no_grad():
            all_preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(all_preds)

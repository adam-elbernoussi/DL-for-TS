"""
MOMENT foundation model pipeline for LSST classification.
M1: Zero-shot embedding extraction + sklearn classifier
M2: Pre-compute encoder features + trainable MLP head (encoder frozen)
M3: Full fine-tuning (end-to-end)

Key insight: MOMENT's ClassificationHead and embed() average across ALL patches
including padding. For short series (36 → 512), 92% of patches are padding.
We fix this by doing masked pooling — only averaging over real-data patches.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from momentfm import MOMENTPipeline
from momentfm.utils.masking import Masking
from sklearn.metrics import f1_score

SEED = 42
PATCH_LEN = 8  # MOMENT's patch length


def _get_patch_mask(input_mask, patch_len=PATCH_LEN):
    """Convert sequence-level mask (batch, 512) to patch-level mask (batch, n_patches).
    A patch is valid only if ALL its timesteps are real (not padding).
    """
    return Masking.convert_seq_to_patch_view(input_mask, patch_len)


# ──────────────────────────────────────────────
# M1: Zero-shot embedding extraction (with masked pooling)
# ──────────────────────────────────────────────

def extract_moment_embeddings(X_padded, input_mask, model_name="AutonLab/MOMENT-1-large",
                               batch_size=64, device="cuda"):
    """Extract MOMENT embeddings with proper masked pooling.
    Instead of model.embed() which averages ALL patches (including padding),
    we run the encoder ourselves and only pool over non-padding patches.

    Returns numpy array of shape (n, n_channels * d_model).
    """
    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={"task_name": "reconstruction"},
    )
    model = model.to(device).float()
    model.eval()

    d_model = model.config.d_model
    all_embeddings = []
    n = X_padded.shape[0]

    for i in range(0, n, batch_size):
        xb = X_padded[i:i+batch_size].to(device).float()
        mb = input_mask[i:i+batch_size].to(device).float()
        bs, n_channels, seq_len = xb.shape

        with torch.no_grad():
            # Normalize
            x_norm = model.normalizer(x=xb, mask=mb, mode="norm")
            x_norm = torch.nan_to_num(x_norm, nan=0, posinf=0, neginf=0)

            # Tokenize + patch embed
            x_tok = model.tokenizer(x=x_norm)
            enc_in = model.patch_embedding(x_tok, mask=mb)
            n_patches = enc_in.shape[2]

            enc_in = enc_in.reshape(bs * n_channels, n_patches, d_model)

            # Build attention mask from input_mask
            patch_mask = _get_patch_mask(mb, PATCH_LEN)  # (bs, n_patches)
            attn_mask = patch_mask.repeat_interleave(n_channels, dim=0)  # (bs*C, n_patches)

            # Encode
            outputs = model.encoder(inputs_embeds=enc_in, attention_mask=attn_mask)
            enc_out = outputs.last_hidden_state  # (bs*C, n_patches, d_model)
            enc_out = enc_out.reshape(bs, n_channels, n_patches, d_model)

            # Masked mean pooling over patches (only real patches)
            # patch_mask: (bs, n_patches) -> (bs, 1, n_patches, 1)
            pm = patch_mask.unsqueeze(1).unsqueeze(-1).float()  # (bs, 1, n_patches, 1)
            masked_sum = (enc_out * pm).sum(dim=2)  # (bs, n_channels, d_model)
            patch_counts = pm.sum(dim=2).clamp(min=1)  # (bs, 1, 1)
            masked_mean = masked_sum / patch_counts  # (bs, n_channels, d_model)

            # Flatten across channels: (bs, n_channels * d_model)
            emb = masked_mean.reshape(bs, n_channels * d_model)

        all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


# ──────────────────────────────────────────────
# M2: Pre-compute features + train standalone head
# ──────────────────────────────────────────────

def extract_moment_classification_features(X_padded, input_mask,
                                            model_name="AutonLab/MOMENT-1-large",
                                            batch_size=64, device="cuda"):
    """Extract per-patch encoder features with masked pooling.
    Returns numpy array of shape (n, n_channels * d_model) — masked mean over patches.
    """
    # Reuse the same logic as extract_moment_embeddings
    return extract_moment_embeddings(
        X_padded, input_mask, model_name=model_name,
        batch_size=batch_size, device=device,
    )


class MOMENTClassificationHead(nn.Module):
    """Standalone classification head to train on pre-computed MOMENT features."""

    def __init__(self, input_dim, n_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        if hidden_dim == 0:
            # Linear-only head (regularized logistic regression)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, n_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )

    def forward(self, x):
        return self.head(x)


def train_classification_head(features_train, y_train, features_val, y_val,
                               class_weights, n_classes=14,
                               epochs=100, lr=1e-3, batch_size=128, patience=15,
                               hidden_dim=256, dropout=0.3, weight_decay=1e-4,
                               device="cuda"):
    """Train a standalone classification head on pre-computed MOMENT features.
    Returns (model, history).
    """
    if features_train.ndim > 2:
        features_train = features_train.reshape(features_train.shape[0], -1)
        features_val = features_val.reshape(features_val.shape[0], -1)

    input_dim = features_train.shape[1]
    model = MOMENTClassificationHead(input_dim, n_classes,
                                     hidden_dim=hidden_dim, dropout=dropout).to(device)

    cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    Xt = torch.tensor(features_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)
    Xv = torch.tensor(features_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)

    trainable = sum(p.numel() for p in model.parameters())
    print(f"Head trainable params: {trainable:,} (input_dim={input_dim})")

    history = {"train_loss": [], "val_loss": [], "val_wf1": []}
    best_wf1 = 0.0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(Xt[idx])
            loss = criterion(logits, yt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        history["train_loss"].append(epoch_loss / n_batches)

        model.eval()
        with torch.no_grad():
            logits_v = model(Xv)
            val_loss = criterion(logits_v, yv).item()
            y_pred_v = logits_v.argmax(1).cpu().numpy()
            wf1 = f1_score(yv.cpu().numpy(), y_pred_v, average="weighted")

        history["val_loss"].append(val_loss)
        history["val_wf1"].append(wf1)

        if wf1 > best_wf1:
            best_wf1 = wf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} — train_loss={history['train_loss'][-1]:.4f}  "
                  f"val_loss={val_loss:.4f}  val_wf1={wf1:.4f}")

    model.load_state_dict(best_state)
    model.to(device)
    return model, history


# ──────────────────────────────────────────────
# M3: Full fine-tuning (end-to-end)
# ──────────────────────────────────────────────

def load_moment_for_classification(n_channels=6, n_classes=14,
                                    model_name="AutonLab/MOMENT-1-large", device="cuda"):
    """Load MOMENT with a classification head, all parameters trainable."""
    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={
            "task_name": "classification",
            "n_channels": n_channels,
            "num_class": n_classes,
        },
    )
    model.init()  # switch head from pretrain to classification
    model = model.to(device).float()

    # Ensure all params are trainable for full fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} params "
          f"({100*trainable/total:.1f}%)")

    return model


def finetune_moment(model, X_train, mask_train, y_train,
                    X_val, mask_val, y_val,
                    class_weights_tensor,
                    epochs=50, lr=1e-3, batch_size=32, patience=10,
                    device="cuda"):
    """Full fine-tune MOMENT for classification (all params trainable).
    Returns (model, history dict).
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(X_train, mask_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": [], "val_wf1": []}
    best_wf1 = 0.0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, mb, yb in train_loader:
            xb, mb, yb = xb.to(device).float(), mb.to(device).float(), yb.to(device)
            out = model(x_enc=xb, input_mask=mb)
            logits = out.logits
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        history["train_loss"].append(epoch_loss / n_batches)

        model.eval()
        with torch.no_grad():
            all_logits = []
            for i in range(0, len(X_val), batch_size):
                xb = X_val[i:i+batch_size].to(device).float()
                mb = mask_val[i:i+batch_size].to(device).float()
                out = model(x_enc=xb, input_mask=mb)
                all_logits.append(out.logits.cpu())
            logits_v = torch.cat(all_logits, dim=0)
            val_loss = criterion(logits_v.to(device), y_val.to(device)).item()
            y_pred_v = logits_v.argmax(1).numpy()
            y_true_v = y_val.cpu().numpy()
            wf1 = f1_score(y_true_v, y_pred_v, average="weighted")

        history["val_loss"].append(val_loss)
        history["val_wf1"].append(wf1)

        if wf1 > best_wf1:
            best_wf1 = wf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}/{epochs} — train_loss={history['train_loss'][-1]:.4f}  "
              f"val_loss={val_loss:.4f}  val_wf1={wf1:.4f}")

    model.load_state_dict(best_state)
    model.to(device)
    return model, history


def predict_moment(model, X, mask, batch_size=64, device="cuda"):
    """Run inference with MOMENT classification model. Returns predicted labels (numpy)."""
    model.eval()
    all_preds = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size].to(device).float()
        mb = mask[i:i+batch_size].to(device).float()
        with torch.no_grad():
            out = model(x_enc=xb, input_mask=mb)
        all_preds.append(out.logits.argmax(1).cpu().numpy())
    return np.concatenate(all_preds)

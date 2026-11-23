import argparse
import os
import random
import pickle
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import CFG
from models import build_model

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TabularDataset(Dataset):
    def __init__(self, df, num_cols, cat_cols, target_col=None):
        self.num_x = torch.tensor(df[num_cols].values, dtype=torch.float32) if num_cols else torch.zeros((len(df), 0))
        self.cat_x = torch.tensor(df[cat_cols].values, dtype=torch.long) if cat_cols else torch.zeros((len(df), 0), dtype=torch.long)
        self.y = torch.tensor(df[target_col].values, dtype=torch.float32) if target_col else None

    def __len__(self):
        return len(self.num_x)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.num_x[idx], self.cat_x[idx], self.y[idx]
        return self.num_x[idx], self.cat_x[idx]

def train_model(model_name, seed):
    seed_everything(seed)
    device = torch.device(CFG.DEVICE)
    tag = model_name if seed == CFG.SEED else f"{model_name}_s{seed}"

    train_df = pd.read_parquet(f"{CFG.ARTIFACT_DIR}/train_proc.parquet")
    test_df = pd.read_parquet(f"{CFG.ARTIFACT_DIR}/test_proc.parquet")
    with open(f"{CFG.ARTIFACT_DIR}/meta.pkl", "rb") as f:
        meta = pickle.load(f)

    num_cols, cat_cols, cat_dims = meta["num_cols"], meta["cat_cols"], meta["cat_dims"]
    target_col = meta["target_col"]

    oof_preds = np.zeros(len(train_df), dtype=np.float64)
    test_preds = np.zeros(len(test_df), dtype=np.float64)

    test_loader = DataLoader(TabularDataset(test_df, num_cols, cat_cols), batch_size=CFG.BATCH_SIZE * 2, shuffle=False)
    scaler = torch.amp.GradScaler("cuda", enabled=CFG.USE_AMP)
    warmup_epochs = 3

    print(f"\n================ BẮT ĐẦU HUẤN LUYỆN: {tag.upper()} ================")
    for fold in range(CFG.N_FOLDS):
        trn_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
        val_idx = train_df["fold"] == fold
        val_df = train_df[val_idx].reset_index(drop=True)

        train_loader = DataLoader(
            TabularDataset(trn_df, num_cols, cat_cols, target_col),
            batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            TabularDataset(val_df, num_cols, cat_cols, target_col),
            batch_size=CFG.BATCH_SIZE * 2, shuffle=False
        )

        model = build_model(model_name, len(num_cols), cat_dims).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, CFG.EPOCHS - warmup_epochs), eta_min=1e-5
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )

        best_auc = 0.0
        best_val_preds = None
        patience_counter = 0
        ckpt_path = f"{CFG.ARTIFACT_DIR}/{tag}_fold{fold}.pt"

        for epoch in range(CFG.EPOCHS):
            model.train()
            for num_x, cat_x, y in train_loader:
                num_x, cat_x, y = num_x.to(device), cat_x.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=CFG.USE_AMP):
                    logits = model(num_x, cat_x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

            model.eval()
            val_probs = []
            with torch.no_grad():
                for num_x, cat_x, _ in val_loader:
                    num_x, cat_x = num_x.to(device), cat_x.to(device)
                    with torch.amp.autocast("cuda", enabled=CFG.USE_AMP):
                        probs = torch.sigmoid(model(num_x, cat_x))
                    val_probs.append(probs.cpu().numpy())
            val_probs = np.concatenate(val_probs).astype(np.float64)
            val_auc = roc_auc_score(val_df[target_col].values, val_probs)

            if val_auc > best_auc:
                best_auc = val_auc
                best_val_preds = val_probs
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= CFG.EARLY_STOPPING_ROUNDS:
                    break

        print(f"[{tag.upper()}] Fold {fold + 1}/{CFG.N_FOLDS} - Best Val AUC: {best_auc:.5f}")
        oof_preds[val_idx] = rankdata(best_val_preds) / len(best_val_preds)

        model.load_state_dict(torch.load(ckpt_path))
        model.eval()
        fold_test = []
        with torch.no_grad():
            for num_x, cat_x in test_loader:
                num_x, cat_x = num_x.to(device), cat_x.to(device)
                with torch.amp.autocast("cuda", enabled=CFG.USE_AMP):
                    probs = torch.sigmoid(model(num_x, cat_x))
                fold_test.append(probs.cpu().numpy())
        fold_test_probs = np.concatenate(fold_test).astype(np.float64)
        test_preds += (rankdata(fold_test_probs) / len(fold_test_probs)) / CFG.N_FOLDS

    overall_auc = roc_auc_score(train_df[target_col].values, oof_preds)
    print(f">>> [{tag.upper()}] OVERALL OOF ROC-AUC: {overall_auc:.5f} <<<\n")

    np.save(f"{CFG.ARTIFACT_DIR}/oof_{tag}.npy", oof_preds)
    np.save(f"{CFG.ARTIFACT_DIR}/test_{tag}.npy", test_preds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["resnet", "swiglu", "fttransformer"])
    parser.add_argument("--seed", type=int, default=CFG.SEED)
    args = parser.parse_args()
    train_model(args.model, args.seed)
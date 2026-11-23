import glob
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import optuna
from config import CFG

optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_ensemble():
    train_df = pd.read_parquet(f"{CFG.ARTIFACT_DIR}/train_proc.parquet")
    test_df = pd.read_parquet(f"{CFG.ARTIFACT_DIR}/test_proc.parquet")
    with open(f"{CFG.ARTIFACT_DIR}/meta.pkl", "rb") as f:
        meta = pickle.load(f)

    target_col = meta["target_col"]
    id_col = meta["id_col"]
    y_true = train_df[target_col].values

    oof_files = sorted(glob.glob(f"{CFG.ARTIFACT_DIR}/oof_*.npy"))
    models = [os.path.basename(p).replace("oof_", "").replace(".npy", "") for p in oof_files]

    oof_dict = {}
    test_dict = {}

    print("--- ĐIỂM OOF TỪNG MÔ HÌNH ĐƠN LẺ ---")
    for m in models:
        oof = np.load(f"{CFG.ARTIFACT_DIR}/oof_{m}.npy")
        tst = np.load(f"{CFG.ARTIFACT_DIR}/test_{m}.npy")
        auc = roc_auc_score(y_true, oof)
        print(f"Model {m:22s} | OOF ROC-AUC: {auc:.5f}")

        oof_dict[m] = rankdata(oof) / len(oof)
        test_dict[m] = rankdata(tst) / len(tst)

    def objective(trial):
        weights = [trial.suggest_float(f"w_{m}", 0.0, 1.0) for m in models]
        if sum(weights) == 0:
            return 0.0
        weights = [w / sum(weights) for w in weights]

        blend_oof = sum(w * oof_dict[m] for w, m in zip(weights, models))
        return roc_auc_score(y_true, blend_oof)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=600)

    best_weights = [study.best_params[f"w_{m}"] for m in models]
    best_weights = [w / sum(best_weights) for w in best_weights]

    print("\n--- KẾT QUẢ ENSEMBLE TỐI ƯU ---")
    for m, w in zip(models, best_weights):
        print(f"Trọng số {m:22s}: {w:.4f}")
    print(f"FINAL ENSEMBLE OOF ROC-AUC: {study.best_value:.5f}")

    final_test_preds = sum(w * test_dict[m] for w, m in zip(best_weights, models))
    sub = pd.DataFrame({
        id_col: test_df[id_col],
        target_col: final_test_preds
    })
    sub.to_csv(CFG.SUB_PATH, index=False)
    print(f"Đã xuất file nộp bài tối ưu tại: {CFG.SUB_PATH}")

if __name__ == "__main__":
    run_ensemble()
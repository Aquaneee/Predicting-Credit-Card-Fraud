import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler, OrdinalEncoder
from config import CFG

warnings.filterwarnings("ignore")

def run_preprocessing():
    print("1. Đang đọc dữ liệu gốc...")
    train_df = pd.read_csv(CFG.TRAIN_PATH)
    test_df = pd.read_csv(CFG.TEST_PATH)

    id_col = "id" if "id" in train_df.columns else train_df.columns[0]
    target_col = [c for c in train_df.columns if c not in test_df.columns][0]
    base_features = [c for c in test_df.columns if c != id_col]

    cat_cols = [
        c for c in base_features
        if train_df[c].dtype == "object" or train_df[c].nunique() < 30
    ]
    num_cols = [c for c in base_features if c not in cat_cols]

    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df[target_col])):
        train_df.loc[val_idx, "fold"] = fold

    print("2. Tạo Frequency Encoding & Group Aggregations...")
    full_df = pd.concat([train_df[base_features], test_df[base_features]], axis=0).reset_index(drop=True)
    new_num_cols = []

    for c in cat_cols:
        freq = full_df[c].value_counts()
        col_name = f"{c}_freq"
        train_df[col_name] = train_df[c].map(freq).fillna(0).astype(np.float32)
        test_df[col_name] = test_df[c].map(freq).fillna(0).astype(np.float32)
        new_num_cols.append(col_name)

    top_cats = [c for c in cat_cols if 2 < train_df[c].nunique() < 50][:3]
    top_nums = num_cols[:5]
    for c in top_cats:
        for n in top_nums:
            grp_mean = full_df.groupby(c)[n].transform("mean")
            train_mean = grp_mean.iloc[:len(train_df)].values
            test_mean = grp_mean.iloc[len(train_df):].values

            diff_col = f"{n}_diff_mean_by_{c}"
            train_df[diff_col] = (train_df[n].values - train_mean).astype(np.float32)
            test_df[diff_col] = (test_df[n].values - test_mean).astype(np.float32)
            new_num_cols.append(diff_col)

    print("3. Tạo Out-Of-Fold Target Encoding...")
    global_mean = train_df[target_col].mean()
    smoothing = 20.0
    for c in cat_cols:
        te_col = f"{c}_oof_te"
        train_df[te_col] = np.nan
        for fold in range(CFG.N_FOLDS):
            trn = train_df[train_df["fold"] != fold]
            stats = trn.groupby(c)[target_col].agg(["count", "mean"])
            smooth_te = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)

            val_mask = train_df["fold"] == fold
            train_df.loc[val_mask, te_col] = train_df.loc[val_mask, c].map(smooth_te).fillna(global_mean)

        full_stats = train_df.groupby(c)[target_col].agg(["count", "mean"])
        full_smooth = (full_stats["count"] * full_stats["mean"] + smoothing * global_mean) / (full_stats["count"] + smoothing)
        test_df[te_col] = test_df[c].map(full_smooth).fillna(global_mean)
        new_num_cols.append(te_col)

    all_num_cols = num_cols + new_num_cols

    print("4. Xử lý Missing & Biến đổi Quantile-Normal cho Neural Networks...")
    for c in all_num_cols:
        med = train_df[c].median()
        train_df[c] = train_df[c].fillna(med).replace([np.inf, -np.inf], med)
        test_df[c] = test_df[c].fillna(med).replace([np.inf, -np.inf], med)

    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=CFG.SEED)
    scaler = StandardScaler()

    train_num_qt = qt.fit_transform(train_df[all_num_cols])
    test_num_qt = qt.transform(test_df[all_num_cols])

    train_df[all_num_cols] = scaler.fit_transform(train_num_qt).astype(np.float32)
    test_df[all_num_cols] = scaler.transform(test_num_qt).astype(np.float32)

    cat_dims = []
    if len(cat_cols) > 0:
        for c in cat_cols:
            train_df[c] = train_df[c].astype(str).fillna("MISSING")
            test_df[c] = test_df[c].astype(str).fillna("MISSING")
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train_df[cat_cols] = oe.fit_transform(train_df[cat_cols]).astype(np.int64) + 1
        test_df[cat_cols] = oe.transform(test_df[cat_cols]).astype(np.int64) + 1
        for c in cat_cols:
            cat_dims.append(int(max(train_df[c].max(), test_df[c].max()) + 1))

    train_df.to_parquet(f"{CFG.ARTIFACT_DIR}/train_proc.parquet", index=False)
    test_df.to_parquet(f"{CFG.ARTIFACT_DIR}/test_proc.parquet", index=False)

    meta = {
        "id_col": id_col,
        "target_col": target_col,
        "num_cols": all_num_cols,
        "cat_cols": cat_cols,
        "cat_dims": cat_dims
    }
    with open(f"{CFG.ARTIFACT_DIR}/meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"Hoàn tất! Tổng số đặc trưng: {len(all_num_cols)} num + {len(cat_cols)} cat.")

if __name__ == "__main__":
    run_preprocessing()
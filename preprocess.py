import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler, OrdinalEncoder
from config import CFG

warnings.filterwarnings("ignore")

def build_clean_features(df):
    df = df.copy()

    bool_cols = [
        "is_foreign_transaction", "is_new_merchant", "used_vpn",
        "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"
    ]
    for c in bool_cols:
        df[c] = df[c].astype(int)

    # 1. Tổng cờ rủi ro & bitmask tổ hợp
    df["total_red_flags"] = df[bool_cols].sum(axis=1).astype(np.float32)
    df["high_risk_combo"] = (
        df["is_foreign_transaction"] * 16 +
        df["used_vpn"] * 8 +
        df["ip_country_mismatch"] * 4 +
        df["billing_shipping_mismatch"] * 2 +
        df["is_ai_generated_scam_attempt"] * 1
    ).astype(str)

    # 2. Tỷ số tài chính & vận tốc vật lý cốt lõi
    df["amount_to_balance_ratio"] = (df["amount_usd"] / (df["account_balance_usd"].clip(lower=0) + 1.0)).astype(np.float32)
    df["travel_speed_kmh"] = (df["distance_from_home_km"] / (df["hours_since_last_txn"].clip(lower=0) + 0.1)).astype(np.float32)
    df["cvv_fail_rate"] = (df["cvv_retry_count"] / (df["txn_count_last_24h"] + 1.0)).astype(np.float32)
    df["risk_x_velocity"] = (df["merchant_risk_score"] * df["velocity_score"]).astype(np.float32)
    df["amount_x_risk"] = (np.log1p(df["amount_usd"].clip(lower=0)) * df["merchant_risk_score"]).astype(np.float32)

    # 3. Chu kỳ thời gian
    df["hour_sin"] = np.sin(2 * np.pi * df["time_of_day_hour"] / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["time_of_day_hour"] / 24.0).astype(np.float32)

    # 4. Cặp tương tác phân loại chọn lọc
    df["inter_merch_channel"] = df["merchant_category"].astype(str) + "_" + df["channel"].astype(str)
    df["inter_auth_device"] = df["auth_method"].astype(str) + "_" + df["device_type"].astype(str)

    return df

def run_preprocessing():
    print("1. Đang đọc dữ liệu gốc...")
    train_df = pd.read_csv(CFG.TRAIN_PATH).dropna(subset=["is_fraud"]).reset_index(drop=True)
    test_df = pd.read_csv(CFG.TEST_PATH)

    id_col, target_col = "id", "is_fraud"
    train_df[target_col] = train_df[target_col].astype(np.float32)

    base_features = [c for c in test_df.columns if c != id_col]
    train_df["row_null_count"] = train_df[base_features].isnull().sum(axis=1).astype(np.float32)
    test_df["row_null_count"] = test_df[base_features].isnull().sum(axis=1).astype(np.float32)

    train_df = build_clean_features(train_df)
    test_df = build_clean_features(test_df)

    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df[target_col])):
        train_df.loc[val_idx, "fold"] = fold

    cat_cols = [
        "merchant_category", "card_type", "auth_method", "channel", "device_type",
        "time_of_day_hour", "day_of_week", "cvv_retry_count", "prior_disputes",
        "high_risk_combo", "inter_merch_channel", "inter_auth_device"
    ]

    num_cols = [
        "amount_usd", "hours_since_last_txn", "txn_count_last_24h",
        "distance_from_home_km", "card_age_months", "customer_age",
        "account_balance_usd", "cvv_retry_count", "velocity_score",
        "merchant_risk_score", "prior_disputes",
        "is_foreign_transaction", "is_new_merchant", "used_vpn",
        "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
        "row_null_count", "total_red_flags", "amount_to_balance_ratio",
        "travel_speed_kmh", "cvv_fail_rate", "risk_x_velocity",
        "amount_x_risk", "hour_sin", "hour_cos"
    ]

    # Chỉ giữ Z-score cho 3 nhóm quan trọng nhất (không tạo OOF Target Encoding gây overfit)
    full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    for grp_col, val_col in [("merchant_category", "amount_usd"), ("card_type", "amount_usd"), ("device_type", "velocity_score")]:
        grp_mean = full_df.groupby(grp_col)[val_col].transform("mean")
        grp_std = full_df.groupby(grp_col)[val_col].transform("std").fillna(1.0)
        z_col = f"{val_col}_z_by_{grp_col}"
        train_df[z_col] = ((train_df[val_col].values - grp_mean.iloc[:len(train_df)].values) / (grp_std.iloc[:len(train_df)].values + 1e-5)).astype(np.float32)
        test_df[z_col] = ((test_df[val_col].values - grp_mean.iloc[len(train_df):].values) / (grp_std.iloc[len(train_df):].values + 1e-5)).astype(np.float32)
        num_cols.append(z_col)

    # Frequency encoding chọn lọc cho 5 cột phân loại chính
    for c in ["merchant_category", "card_type", "auth_method", "channel", "device_type"]:
        freq = full_df[c].astype(str).value_counts()
        f_col = f"{c}_freq"
        train_df[f_col] = train_df[c].astype(str).map(freq).fillna(0).astype(np.float32)
        test_df[f_col] = test_df[c].astype(str).map(freq).fillna(0).astype(np.float32)
        num_cols.append(f_col)

    for c in num_cols:
        med = train_df[c].median()
        train_df[c] = train_df[c].fillna(med).replace([np.inf, -np.inf], med)
        test_df[c] = test_df[c].fillna(med).replace([np.inf, -np.inf], med)

    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=CFG.SEED)
    scaler = StandardScaler()

    train_df[num_cols] = scaler.fit_transform(qt.fit_transform(train_df[num_cols])).astype(np.float32)
    test_df[num_cols] = scaler.transform(qt.transform(test_df[num_cols])).astype(np.float32)

    cat_dims = []
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

    with open(f"{CFG.ARTIFACT_DIR}/meta.pkl", "wb") as f:
        pickle.dump({"id_col": id_col, "target_col": target_col, "num_cols": num_cols, "cat_cols": cat_cols, "cat_dims": cat_dims}, f)

    print(f"Hoàn tất! Bộ đặc trưng tinh gọn: {len(num_cols)} cột số + {len(cat_cols)} cột Embedding.")

if __name__ == "__main__":
    run_preprocessing()
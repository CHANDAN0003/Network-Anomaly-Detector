import pandas as pd
import json
import os

# -------------------------------
# Vocabulary mapping for top correlated features
# -------------------------------
VOCAB = {
    "syn_flag": {0: "SYN_0", 1: "SYN_1", 2: "SYN_2"},  # Other values handled as SYN_OTHER
    "fwd_pkts_s": ["FPKT_LOW", "FPKT_MED", "FPKT_HIGH"],
    "flow_pkts_s": ["FLPKT_LOW", "FLPKT_MED", "FLPKT_HIGH"],
    "bwd_pkts_s": ["BPKT_LOW", "BPKT_MED", "BPKT_HIGH"],
    "init_bwd_win": ["WIN_LOW", "WIN_MED", "WIN_HIGH"],
    "active_min": ["ACTMIN_LOW", "ACTMIN_MED", "ACTMIN_HIGH"],
    "active_mean": ["ACTMEAN_LOW", "ACTMEAN_MED", "ACTMEAN_HIGH"],
    "fwd_pkt_len_min": ["FPLEN_LOW", "FPLEN_MED", "FPLEN_HIGH"],
    "bwd_iat_tot": ["BIAT_LOW", "BIAT_MED", "BIAT_HIGH"]
}

# -------------------------------
# Helper: Generic binning
# -------------------------------
def bin_feature(x, bins, labels):
    """Assigns a token label to a numeric value based on given bins."""
    try:
        return pd.cut([x], bins=bins, labels=labels, include_lowest=True)[0]
    except Exception:
        return None

# -------------------------------
# Convert a single flow into a DNA-like token sequence
# -------------------------------
def get_token_for_flow(row):
    tokens = []
    # SYN Flag Cnt (categorical mapping)
    tokens.append(VOCAB["syn_flag"].get(int(row.get("SYN Flag Cnt", 0)), "SYN_OTHER"))
    # Continuous features → bins
    tokens.append(bin_feature(row.get("Fwd Pkts/s", 0), [0, 50, 200, float('inf')], VOCAB["fwd_pkts_s"]))
    tokens.append(bin_feature(row.get("Flow Pkts/s", 0), [0, 50, 200, float('inf')], VOCAB["flow_pkts_s"]))
    tokens.append(bin_feature(row.get("Bwd Pkts/s", 0), [0, 50, 200, float('inf')], VOCAB["bwd_pkts_s"]))
    tokens.append(bin_feature(row.get("Init Bwd Win Byts", 0), [0, 1000, 5000, float('inf')], VOCAB["init_bwd_win"]))
    tokens.append(bin_feature(row.get("Active Min", 0), [0, 100, 500, float('inf')], VOCAB["active_min"]))
    tokens.append(bin_feature(row.get("Active Mean", 0), [0, 100, 500, float('inf')], VOCAB["active_mean"]))
    tokens.append(bin_feature(row.get("Fwd Pkt Len Min", 0), [0, 50, 200, float('inf')], VOCAB["fwd_pkt_len_min"]))
    tokens.append(bin_feature(row.get("Bwd IAT Tot", 0), [0, 1000, 5000, float('inf')], VOCAB["bwd_iat_tot"]))
    # Join into DNA-like sequence (skip None values)
    return " ".join(str(t) for t in tokens if pd.notna(t))

# -------------------------------
# Main processing function
# -------------------------------
def process_ctu13(attack_csv, normal_csv, base_path, top_k=10):
    # Paths for saving output files
    results_path = os.path.join(base_path, "results")
    data_processed_path = os.path.join(base_path, "data", "processed")
    tokenizer_config_path = os.path.join(base_path, "data", "tokenizer_config.json")
    output_csv = os.path.join(data_processed_path, "tokenized.csv")

    # Create directories if they don't exist
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(data_processed_path, exist_ok=True)

    # --- Load and label data ---
    df_attack = pd.read_csv(attack_csv)
    df_attack['Label'] = 1

    df_normal = pd.read_csv(normal_csv)
    df_normal['Label'] = 0

    # --- Merge ---
    df = pd.concat([df_attack, df_normal], ignore_index=True)
    print(f"[INFO] Combined dataset shape: {df.shape}")

    # --- Correlation with Label ---
    numeric_df = df.select_dtypes(include=['number'])
    if 'Label' in numeric_df.columns:
        corr = numeric_df.corr()['Label'].abs().sort_values(ascending=False)
        print("\n[INFO] Top correlated features with Label:")
        print(corr.head(top_k))
        corr.to_csv(os.path.join(results_path, "top_correlated_features.csv"))
        print(f"[INFO] Saved top correlated features → {os.path.join(results_path, 'top_correlated_features.csv')}")
    else:
        print("[WARNING] No 'Label' column found for correlation analysis!")

    # --- Tokenize flows ---
    sequences = [get_token_for_flow(row) for _, row in df.iterrows()]
    labels = df['Label'].tolist()

    processed_df = pd.DataFrame({"sequence": sequences, "label": labels})
    processed_df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved tokenized data → {output_csv}")

    # --- Save tokenizer config ---
    with open(tokenizer_config_path, "w") as f:
        json.dump(VOCAB, f, indent=4)
    print(f"[INFO] Saved tokenizer config → {tokenizer_config_path}")

# -------------------------------
# Run as standalone script
# -------------------------------
if __name__ == "__main__":
    base_path = r"D:\Network-Anomaly-Detector"
    attack_path = os.path.join(base_path, "CTU13-CSV-Dataset-main", "CTU13_Attack_Traffic.csv")
    normal_path = os.path.join(base_path, "CTU13-CSV-Dataset-main", "CTU13_Normal_Traffic.csv")
    process_ctu13(attack_path, normal_path, base_path)

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def save_training_logs(base_path, loss_per_epoch, eval_metrics):
    results_path = os.path.join(base_path, "results")
    os.makedirs(results_path, exist_ok=True)

    # Save loss graph
    if loss_per_epoch:
        plt.figure()
        plt.plot(range(1, len(loss_per_epoch)+1), loss_per_epoch, marker='o')
        plt.title("Training Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        loss_graph_path = os.path.join(results_path, "training_loss.png")
        plt.savefig(loss_graph_path)
        plt.close()
        print(f"[INFO] Saved loss graph → {loss_graph_path}")

        # Save training logs as CSV
        log_csv_path = os.path.join(results_path, "training_logs.csv")
        df_logs = pd.DataFrame({"epoch": list(range(1, len(loss_per_epoch)+1)),
                                "loss": loss_per_epoch})
        df_logs.to_csv(log_csv_path, index=False)
        print(f"[INFO] Saved training logs → {log_csv_path}")

    # Save evaluation metrics as JSON
    eval_json_path = os.path.join(results_path, "evaluation_metrics.json")
    with open(eval_json_path, "w") as f:
        json.dump(eval_metrics, f, indent=4)
    print(f"[INFO] Saved evaluation metrics → {eval_json_path}")

def load_training_logs(base_path):
    results_path = os.path.join(base_path, "results")
    log_csv_path = os.path.join(results_path, "training_logs.csv")
    eval_json_path = os.path.join(results_path, "evaluation_metrics.json")

    logs = None
    metrics = None

    if os.path.exists(log_csv_path):
        logs = pd.read_csv(log_csv_path)
        print(f"[INFO] Loaded training logs from {log_csv_path}")
    else:
        print(f"[WARN] Training logs CSV not found at {log_csv_path}")

    if os.path.exists(eval_json_path):
        with open(eval_json_path, "r") as f:
            metrics = json.load(f)
        print(f"[INFO] Loaded evaluation metrics from {eval_json_path}")
    else:
        print(f"[WARN] Evaluation metrics JSON not found at {eval_json_path}")

    return logs, metrics

if __name__ == "__main__":
    # Example data for testing
    base_path = "."
    loss_per_epoch = [0.9, 0.7, 0.5, 0.4, 0.3]
    eval_metrics = {"accuracy": 0.95, "precision": 0.9, "recall": 0.92, "f1": 0.91}
    save_training_logs(base_path, loss_per_epoch, eval_metrics)

    # Example usage for loading and printing
    logs, metrics = load_training_logs(base_path)
    if logs is not None:
        print("\n[INFO] Training Logs:\n", logs)
    if metrics is not None:
        print("\n[INFO] Evaluation Metrics:\n", metrics)
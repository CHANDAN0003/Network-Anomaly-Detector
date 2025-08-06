import sys, os
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from train_model import DNADataset, LSTMClassifier, collate_fn
from results.save_results import save_training_logs  # <-- Import your logger

def evaluate(csv_path, model_path, vocab_path, base_path, loss_per_epoch=None):
    # Load processed dataset
    df = pd.read_csv(csv_path)

    # Load the saved vocab to ensure correct model size
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    test_ds = DNADataset(df['sequence'].tolist(), df['label'].tolist(), vocab)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Load model with correct vocab size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(len(vocab)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Collect predictions
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            pred = out.argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n[INFO] Evaluation Metrics:")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nDetailed Report:\n", classification_report(y_true, y_pred, zero_division=0))

    # Save results if loss_per_epoch is provided (from training)
    eval_metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }
    if loss_per_epoch is not None:
        save_training_logs(
            base_path=base_path,
            loss_per_epoch=loss_per_epoch,
            eval_metrics=eval_metrics
        )
    else:
        print("[INFO] Skipped saving training logs because loss_per_epoch was not provided.")

if __name__ == "__main__":
    # If only evaluating (no training losses), pass loss_per_epoch=None
    evaluate(
        csv_path=r"D:\Network-Anomaly-Detector\data\processed\tokenized.csv",
        model_path=r"D:\Network-Anomaly-Detector\model\saved_model\dna_lstm.pth",
        vocab_path=r"D:\Network-Anomaly-Detector\model\saved_model\vocab.pkl",
        base_path=r"D:\Network-Anomaly-Detector",
        loss_per_epoch=None  # Replace with actual list if you have it from training
    )

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
from results.save_results import save_training_logs  # <-- Import logger here
import pickle
# -----------------------
# Dataset class
# -----------------------
class DNADataset(Dataset):
    def __init__(self, sequences, labels, vocab=None):
        tokens = sorted(set(" ".join(sequences).split()))
        self.vocab = {tok: i+1 for i, tok in enumerate(tokens)} if vocab is None else vocab
        self.sequences = [self.encode(seq) for seq in sequences]
        self.labels = torch.tensor(labels).long()
    
    def encode(self, seq):
        return torch.tensor([self.vocab.get(tok, 0) for tok in seq.split()])
    
    def __len__(self): 
        return len(self.sequences)
    
    def __getitem__(self, idx): 
        return self.sequences[idx], self.labels[idx]

# -----------------------
# Model
# -----------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size+1, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# -----------------------
# Collate function
# -----------------------
def collate_fn(batch):
    seqs, labels = zip(*batch)
    seqs_pad = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    return seqs_pad, torch.tensor(labels)

# -----------------------
# Evaluation helper
# -----------------------
def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            preds = torch.argmax(out, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("\n[INFO] Evaluation Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nDetailed Report:\n", classification_report(y_true, y_pred, zero_division=0))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# -----------------------
# Training function
# -----------------------
def train_model(csv_path, model_path, base_path):
    df = pd.read_csv(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df['sequence'], df['label'], test_size=0.2, random_state=42)
    
    train_ds = DNADataset(X_train.tolist(), y_train.tolist(), None)
    test_ds  = DNADataset(X_test.tolist(), y_test.tolist(), train_ds.vocab)
    vocab_size = len(train_ds.vocab)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(vocab_size).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_per_epoch = []

    for epoch in range(5):
        model.train()
        total_loss = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            out = model(X)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_per_epoch.append(avg_loss)
        print(f"[INFO] Epoch {epoch+1} Loss: {avg_loss:.4f}")


# Save model
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Saved trained model → {model_path}")

    # Save vocab
    vocab_path = r"D:\Network-Anomaly-Detector\model\saved_model\vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(train_ds.vocab, f)
    print(f"[INFO] Saved vocab → {vocab_path}")


    eval_metrics = evaluate_model(model, test_loader, device)
    save_training_logs(base_path, loss_per_epoch, eval_metrics)

# -----------------------
# Run training
# -----------------------
if __name__ == "__main__":
    train_model(
        r"D:\Network-Anomaly-Detector\data\processed\tokenized.csv", 
        r"D:\Network-Anomaly-Detector\model\saved_model\dna_lstm.pth",
        base_path=r"D:\Network-Anomaly-Detector"
    )

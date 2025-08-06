import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.train_model import LSTMClassifier, DNADataset
import streamlit as st
import torch
import json
import pickle


# -----------------------
# Load tokenizer vocab
# -----------------------
with open(r"D:\Network-Anomaly-Detector\data\tokenizer_config.json", "r") as f:
    vocab_config = json.load(f)

# Load the saved vocab
with open(r"D:\Network-Anomaly-Detector\model\saved_model\vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Initialize the model with the correct vocab size
model = LSTMClassifier(len(vocab))

# Now load the weights
model.load_state_dict(torch.load(r"D:\Network-Anomaly-Detector\model\saved_model\dna_lstm.pth", map_location="cpu"))


# -----------------------
# Streamlit UI
# -----------------------
st.title("Network DNA Anomaly Detector")
user_seq = st.text_input(
    "Enter a network DNA sequence (tokens separated by spaces):",
    "SYN_1 FPKT_LOW FLPKT_LOW BPKT_LOW WIN_LOW ACTMIN_LOW ACTMEAN_LOW FPLEN_LOW BIAT_LOW"
)

if st.button("Predict"):
    # Use existing vocabulary from training
    dataset = DNADataset([user_seq], [0], vocab={tok:i+1 for i, tok in enumerate(vocab_config)})
    seq_tensor = dataset[0][0].unsqueeze(0)  # shape [1, seq_len]

    # Inference
    with torch.no_grad():
        out = model(seq_tensor)
        pred = out.argmax(dim=1).item()
    
    st.write("### Prediction:", "🚨 **Anomalous**" if pred == 1 else "✅ **Normal**")

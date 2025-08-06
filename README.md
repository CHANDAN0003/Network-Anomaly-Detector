# Network DNA Fingerprinting for Encrypted Traffic Anomaly Detection

This project detects anomalies in encrypted network traffic by converting flow metadata into **pseudo-DNA sequences** and classifying them using **deep learning models** (LSTM/Transformer).  
The system uses the **CTU-13 dataset** for training and evaluation.


## Features
- Converts network flow features into tokenized sequences
- Supports LSTM/Transformer for classification
- Interactive **Streamlit dashboard** for visualization and testing
- Supports real-time traffic analysis (planned extension)


## Project Structure
├── dashboard/
│ └── streamlit_app.py # Main dashboard app
├── model/
│ ├── train_model.py # Training script
│ ├── evaluate.py # Evaluation script
├── nlp_encoder/
│ └── flow_to_dna.py # Converts flow metadata to DNA sequences
├── results/
│ ├── save_results.py # Saves model outputs
│ └── init.py
├── Trial.ipynb # Jupyter notebook for experiments
├── requirements.txt # Dependencies
├── README.md # Project documentation


## Installation

### 1. Clone the repository
git clone github.com/CHANDAN0003/Network-Anomaly-Detector.git
cd Network-Anomaly-Detector

### 2. Set up a virtual environment 
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate

### 3.Install dependencies
pip install -r requirements.txt





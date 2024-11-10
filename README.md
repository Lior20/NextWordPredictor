# 🔮 RNN Next-Word Prediction on Penn Tree Bank Dataset

This project implements various configurations of recurrent neural networks (RNNs) for next-word prediction on the Penn Tree Bank dataset. The goal is to achieve perplexities below 125 (without dropout) and 100 (with dropout).

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)


## 📋 Project Overview

The implementation includes:
- LSTM and GRU architectures
- Configurable dropout for regularization
- Detailed performance tracking and visualization
- Comprehensive evaluation metrics

## 🛠️ Dependencies

```
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.4.0
tqdm>=4.65.0  # Optional, for progress bars
```

## 📦 Installation & Setup

1. **Dataset Preparation**
   - Download the Penn Tree Bank dataset from your course Moodle
   - Extract files to your specified base path
   - Ensure the following files are present:
     - `ptb.train.txt`
     - `ptb.valid.txt`
     - `ptb.test.txt`

2. **Environment Setup**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Model Configuration

Key parameters that can be modified:
```python
embd_dim = 128      # Embedding dimension
n_hidden = 200      # Hidden layer size
n_layers = 2        # Number of RNN layers
seq_len = 20        # Sequence length
learning_rate = 0.001
dropout = 0.5       # Set to 0 for no dropout
batch_size = 128
is_lstm = True      # True for LSTM, False for GRU
```

### Training

```python
python next_word_prediction.py
```

## 🏗️ Architecture

The project consists of three main components:

1. **Data Processing (`PTBDataset` class)**
   - Handles data loading and preprocessing
   - Manages vocabulary creation
   - Creates training, validation, and test batches

2. **Model Architecture (`NextWordPredict` class)**
   - Configurable RNN type (LSTM/GRU)
   - Embedding layer
   - Optional dropout
   - Linear output layer

3. **Training & Evaluation**
   - Early stopping based on validation perplexity
   - Performance visualization
   - Comprehensive testing framework

## 📊 Results & Visualization

The training process generates:
- Convergence graphs for each configuration
- Perplexity metrics for train/validation/test sets
- Performance comparison between architectures

## 🔍 Model Configurations

The project includes four main configurations:
1. LSTM without dropout
2. LSTM with dropout
3. GRU without dropout
4. GRU with dropout

## 💾 Pre-trained Models

To use pre-trained models:
1. Set the appropriate model path
2. Configure the model parameters to match the pre-trained model
3. Load and evaluate using the provided functions

## 📈 Performance Monitoring

The system tracks:
- Training loss
- Validation perplexity
- Test perplexity
- Convergence metrics

## ⚡ Optimization Features

- Gradient clipping for stability
- Adam optimizer with configurable learning rate
- Early stopping mechanism
- Batch processing for efficiency

## 📝 License

This project is part of an academic course and should be used according to your institution's academic integrity policies.

"""
BehaviorGuard - Sequence-Aware Detection + Classification Model
===================================================================
Deliverables 3 & 4:
  3. Detection model: sequence-aware (LSTM) approach that flags deviations.
  4. Anomaly classification: not just "anomalous", but which attack
     category it resembles.

A single small LSTM encodes the last SEQ_LEN behavioral-deviation
vectors for an entity. Two heads sit on top of the final hidden state:
  - risk head:  P(this event is NOT normal)      -> the analyst risk score
  - class head: softmax over attack categories    -> the anomaly type

Both heads share the encoder, which keeps the model small and fast to
train -- appropriate for a near-real-time streaming setting.
"""
import torch
import torch.nn as nn


class BehaviorLSTM(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.class_head = nn.Linear(hidden_size, n_classes)
        # risk_head reuses class_head's "normal" logit implicitly, but we
        # keep a dedicated binary head so risk score is directly calibratable
        self.risk_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, F)
        out, (h_n, c_n) = self.lstm(x)
        h = self.norm(h_n[-1])          # (B, hidden)
        h = self.dropout(h)
        class_logits = self.class_head(h)      # (B, n_classes)
        risk_logit = self.risk_head(h).squeeze(-1)  # (B,)
        return class_logits, risk_logit

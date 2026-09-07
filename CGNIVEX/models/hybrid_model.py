import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from config import EMBEDDING_DIM, GNN_EMBEDDING_DIM, EMOTION_LABELS, TOPIC_LABELS, RISK_LEVELS

class PyTorchMLPClassifier(nn.Module):
    """
    Multi-layer Perceptron head for multi-task prediction from combined embeddings.
    """
    def __init__(self, in_dim: int, num_emotions: int, num_topics: int, num_risks: int):
        super(PyTorchMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)

        # Multi-task output heads
        self.emotion_head = nn.Linear(64, num_emotions)
        self.topic_head = nn.Linear(64, num_topics)
        self.risk_head = nn.Linear(64, num_risks)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.dropout(h)
        h = self.relu(self.fc2(h))

        emotion_logits = self.emotion_head(h)
        topic_logits = self.topic_head(h)
        risk_logits = self.risk_head(h)
        return emotion_logits, topic_logits, risk_logits


class CGNIVEXHybridModel:
    """
    Proposed CGNIVEX Core Hybrid Model Architecture.
    Combines Contextual Transformer Text Embeddings + Graph Neural Network Structural Embeddings
    into a joint representation for multi-task Mental Health Risk Indicator prediction.
    """

    def __init__(self):
        self.input_dim = EMBEDDING_DIM + GNN_EMBEDDING_DIM
        self.emotion_encoder = LabelEncoder().fit(EMOTION_LABELS)
        self.topic_encoder = LabelEncoder().fit(TOPIC_LABELS)
        self.risk_encoder = LabelEncoder().fit(RISK_LEVELS)

        self.mlp = PyTorchMLPClassifier(
            in_dim=self.input_dim,
            num_emotions=len(EMOTION_LABELS),
            num_topics=len(TOPIC_LABELS),
            num_risks=len(RISK_LEVELS)
        )
        self.is_trained = False

    def concatenate_embeddings(self, text_emb: np.ndarray, graph_emb: np.ndarray) -> np.ndarray:
        """
        Concatenates text embedding and graph embedding vectors.
        """
        if text_emb.ndim == 1:
            text_emb = text_emb.reshape(1, -1)
        if graph_emb.ndim == 1:
            graph_emb = graph_emb.reshape(1, -1)

        # Truncate or pad to match required dimensions
        if text_emb.shape[1] != EMBEDDING_DIM:
            text_emb = np.pad(text_emb, ((0, 0), (0, max(0, EMBEDDING_DIM - text_emb.shape[1]))))[:, :EMBEDDING_DIM]
        if graph_emb.shape[1] != GNN_EMBEDDING_DIM:
            graph_emb = np.pad(graph_emb, ((0, 0), (0, max(0, GNN_EMBEDDING_DIM - graph_emb.shape[1]))))[:, :GNN_EMBEDDING_DIM]

        return np.hstack([text_emb, graph_emb])

    def fit(self, X_text: np.ndarray, X_graph: np.ndarray, y_emotion: list, y_topic: list, y_risk: list, epochs: int = 15):
        """
        Trains the CGNIVEX Hybrid MLP model multi-task classifier.
        """
        X_combined = self.concatenate_embeddings(X_text, X_graph)
        inputs_t = torch.tensor(X_combined, dtype=torch.float32)

        # Handle unknown labels gracefully
        y_e_idx = [self.emotion_encoder.transform([e])[0] if e in self.emotion_encoder.classes_ else 0 for e in y_emotion]
        y_t_idx = [self.topic_encoder.transform([t])[0] if t in self.topic_encoder.classes_ else 0 for t in y_topic]
        y_r_idx = [self.risk_encoder.transform([r])[0] if r in self.risk_encoder.classes_ else 0 for r in y_risk]

        target_e = torch.tensor(y_e_idx, dtype=torch.long)
        target_t = torch.tensor(y_t_idx, dtype=torch.long)
        target_r = torch.tensor(y_r_idx, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.mlp.parameters(), lr=0.005)

        self.mlp.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out_e, out_t, out_r = self.mlp(inputs_t)
            loss_e = criterion(out_e, target_e)
            loss_t = criterion(out_t, target_t)
            loss_r = criterion(out_r, target_r)
            total_loss = loss_e + loss_t + loss_r
            total_loss.backward()
            optimizer.step()

        self.is_trained = True
        print(f"[CGNIVEXHybridModel] Completed training for {epochs} epochs. Final Loss: {total_loss.item():.4f}")

    def predict(self, text_emb: np.ndarray, graph_emb: np.ndarray) -> dict:
        """
        Performs multi-task inference returning predicted Emotion, Topic, and Mental Health Risk Indicator.
        """
        X_combined = self.concatenate_embeddings(text_emb, graph_emb)
        inputs_t = torch.tensor(X_combined, dtype=torch.float32)

        self.mlp.eval()
        with torch.no_grad():
            out_e, out_t, out_r = self.mlp(inputs_t)
            idx_e = torch.argmax(out_e, dim=1).item()
            idx_t = torch.argmax(out_t, dim=1).item()
            idx_r = torch.argmax(out_r, dim=1).item()

        return {
            "predicted_emotion": self.emotion_encoder.inverse_transform([idx_e])[0],
            "predicted_topic": self.topic_encoder.inverse_transform([idx_t])[0],
            "risk_indicator": self.risk_encoder.inverse_transform([idx_r])[0],
            "combined_embedding": X_combined
        }

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.transformer_model import TransformerEmbeddingModel
from models.hybrid_model import CGNIVEXHybridModel

class TestModels(unittest.TestCase):

    def test_transformer_model(self):
        model = TransformerEmbeddingModel(use_transformer=False)
        emb = model.generate_embedding(["feeling very stressed about exams"])
        self.assertEqual(emb.shape[0], 1)
        self.assertEqual(emb.shape[1], 128)

    def test_hybrid_model_concat(self):
        hybrid = CGNIVEXHybridModel()
        t_emb = np.random.randn(1, 128)
        g_emb = np.random.randn(1, 64)
        combined = hybrid.concatenate_embeddings(t_emb, g_emb)
        self.assertEqual(combined.shape, (1, 192))

if __name__ == "__main__":
    unittest.main()

import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph.graph_builder import ContextualGraphBuilder

class TestGraph(unittest.TestCase):

    def test_graph_construction(self):
        df = pd.DataFrame([{
            "post_id": "P101",
            "user_id": "usr_001",
            "topic": "Academic pressure",
            "label": "Anxiety/Stress",
            "cleaned_text": "stressed exam study deadline"
        }])

        builder = ContextualGraphBuilder()
        g = builder.build_graph(df)

        self.assertGreater(g.number_of_nodes(), 0)
        self.assertTrue(g.has_node("user_usr_001"))
        self.assertTrue(g.has_node("post_P101"))
        self.assertTrue(g.has_node("topic_Academic pressure"))

if __name__ == "__main__":
    unittest.main()

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GNN_EMBEDDING_DIM

# Check PyTorch Geometric Availability
PYG_AVAILABLE = False
try:
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except Exception:
    PYG_AVAILABLE = False


class PyGGraphSAGE(nn.Module):
    """
    PyTorch Geometric 2-layer GraphSAGE architecture for neighbor aggregation
    and node representation learning.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(PyGGraphSAGE, self).__init__()
        if PYG_AVAILABLE:
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        if not PYG_AVAILABLE:
            return x
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GNNRepresentationModel:
    """
    Graph Neural Representation Learning Module for CGNIVEX.
    Learns structural embeddings of users/posts/topics from graph topology.
    Uses GraphSAGE via PyTorch Geometric if available, or NetworkX adjacency fallback.
    """

    def __init__(self, embedding_dim: int = GNN_EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.pyg_model = None
        self.node_mapping = {}

    def extract_graph_embeddings(self, graph: nx.Graph) -> dict:
        """
        Generates graph embeddings for all nodes in the input graph.
        Returns a dictionary mapping node_id -> embedding_vector (shape: (GNN_EMBEDDING_DIM,)).
        """
        if graph.number_of_nodes() == 0:
            return {}

        nodes = list(graph.nodes())
        self.node_mapping = {n: i for i, n in enumerate(nodes)}
        num_nodes = len(nodes)

        if PYG_AVAILABLE and num_nodes > 1:
            try:
                # 1. Create simple identity feature matrix
                x = torch.eye(num_nodes, dtype=torch.float)
                
                # 2. Extract edge index array
                edges = list(graph.edges())
                if edges:
                    edge_index_list = []
                    for u, v in edges:
                        edge_index_list.append([self.node_mapping[u], self.node_mapping[v]])
                        edge_index_list.append([self.node_mapping[v], self.node_mapping[u]])
                    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)

                # 3. Instantiate and run PyG GraphSAGE
                sage_net = PyGGraphSAGE(in_channels=num_nodes, hidden_channels=32, out_channels=self.embedding_dim)
                sage_net.eval()
                with torch.no_grad():
                    out_embeddings = sage_net(x, edge_index).numpy()

                embeddings_dict = {n: out_embeddings[i] for n, i in enumerate(nodes)}
                print(f"[GNNModel] Successfully computed PyTorch Geometric GraphSAGE embeddings for {num_nodes} nodes.")
                return embeddings_dict
            except Exception as e:
                print(f"[GNNModel] PyG GraphSAGE execution warning ({e}). Using NetworkX adjacency fallback.")

        # Fallback NetworkX Adjacency / Spectral Embedding
        embeddings_dict = {}
        adj_matrix = nx.to_numpy_array(graph)
        
        # Simple SVD/Principal Component reduction on Adjacency Matrix
        try:
            U, S, Vt = np.linalg.svd(adj_matrix)
            reduced = U[:, :self.embedding_dim]
            if reduced.shape[1] < self.embedding_dim:
                padding = np.zeros((num_nodes, self.embedding_dim - reduced.shape[1]))
                reduced = np.hstack([reduced, padding])
        except Exception:
            np.random.seed(42)
            reduced = np.random.randn(num_nodes, self.embedding_dim) * 0.1

        for idx, node in enumerate(nodes):
            embeddings_dict[node] = reduced[idx]

        print(f"[GNNModel] NetworkX adjacency embeddings computed for {num_nodes} nodes.")
        return embeddings_dict

    def get_post_graph_embedding(self, graph: nx.Graph, post_id: str, embeddings_dict: dict) -> np.ndarray:
        """
        Retrieves or computes graph embedding vector for a given post.
        """
        target_key = f"post_{post_id}" if not post_id.startswith("post_") else post_id
        if target_key in embeddings_dict:
            return embeddings_dict[target_key]
        
        # Default mean embedding if post node missing
        if embeddings_dict:
            return np.mean(list(embeddings_dict.values()), axis=0)
        return np.zeros(self.embedding_dim)

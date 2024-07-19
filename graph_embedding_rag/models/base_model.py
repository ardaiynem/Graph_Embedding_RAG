from abc import ABC, abstractmethod

class BaseGraphEmbeddingModel(ABC):
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    @abstractmethod
    def create_model(self, edge_index, device):
        pass

    @abstractmethod
    def train_model(self, model, device):
        pass

    @abstractmethod
    def generate_embeddings(self, model, device, graph):
        pass
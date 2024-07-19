from .base_model import BaseGraphEmbeddingModel
import torch
from torch_geometric.nn import DeepGraphInfomax, GCNConv

class DeepGraphInfomaxModel(BaseGraphEmbeddingModel):
    def create_model(self, edge_index, device):
        # This is a simplified example and might need adjustment
        from torch_geometric.nn import GCNConv

        class Encoder(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels):
                super().__init__()
                self.conv = GCNConv(in_channels, hidden_channels, cached=True)
                self.prelu = torch.nn.PReLU(hidden_channels)

            def forward(self, x, edge_index):
                return self.prelu(self.conv(x, edge_index))

        num_features = edge_index.max().item() + 1  # Number of nodes
        encoder = Encoder(num_features, self.embedding_dim)
        return DeepGraphInfomax(
            hidden_channels=self.embedding_dim,
            encoder=encoder,
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=lambda x, *args, **kwargs: x + torch.randn_like(x),
        ).to(device)

    def train_model(self, model, device):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        x = torch.randn((model.encoder.conv.in_channels, model.hidden_channels)).to(
            device
        )

        model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            pos_z, neg_z, summary = model(x, model.encoder.conv.cached_edge_index)
            loss = model.loss(pos_z, neg_z, summary)
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch:03d}, Loss: {loss.item():.4f}")

    def generate_embeddings(self, model, device, graph):
        model.eval()
        embeddings = {}
        with torch.no_grad():
            x = torch.randn((model.encoder.conv.in_channels, model.hidden_channels)).to(
                device
            )
            z, _, _ = model(x, model.encoder.conv.cached_edge_index)
            for i, node in enumerate(graph.nodes()):
                embeddings[node] = z[i].cpu().numpy()
        return embeddings
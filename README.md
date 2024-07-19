# Graph Embedding RAG Project

## Overview

The Graph Embedding RAG (Retriever-Augmented Generation) project leverages graph embeddings and large language models to enhance question answering and text generation capabilities. By embedding nodes of a graph into a continuous vector space and using a retriever, we can improve the context relevance and accuracy of generated responses.

## Tech Stack

- **Neo4j**: A graph database used to store and manage graph data.
- **PyTorch Geometric**: A library for deep learning on graphs, used for creating and training graph neural network models.
- **Node2Vec**: A graph embedding algorithm used to generate node embeddings.
- **DeepGraphInfomax**: Another graph embedding method based on mutual information maximization.
- **LlamaIndex**: A framework used to integrate with large language models (LLMs) and manage embeddings.
- **Ollama**: A service to host and utilize large language models (LLMs).

## Installation

Ensure you have Python installed. Then, install the required packages:

```bash
pip install torch torch-geometric neo4j networkx llama-index tqdm
```

## Usage

```python
# Initialize RAG object
rag_node2vec = GraphEmbeddingRAG(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, model_class=Node2VecModel)

# 1: Fetch neo4j graph data and consturct a local networkx graph
rag_node2vec.fetch_graph_data()

# 2: Create embeddings from the knowledge graph by the choosen method (Node2Vec in this case)
rag_node2vec.create_embeddings()

# 3: Add embeddings to the vector DB (Stored in neo4j in this case)
rag_node2vec.add_to_vector_store()

# 4: Initialize the embedding model, index and query engine for llamaindex
rag_node2vec.init_index_and_query_engine()

# Run an example prompt
rag_node2vec.run_prompt("Who follows Tina Guerrero ?")
```

## Supported Models

- Node2Vec: Generates embeddings by simulating random walks over the graph.
- DeepGraphInfomax: Leverages mutual information maximization for unsupervised graph representation learning.

## Features

- Fetch graph data from Neo4j
- Generate embeddings using Node2Vec and DeepGraphInfomax
- Store embeddings into a vector store
- Integration with large language models for enhanced query responses
- Build index and query engine for prompt-based interactions

Thanks to the developers of Neo4j, PyTorch Geometric, and LlamaIndex for providing the foundational tools used in this project.
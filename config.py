from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

model_name = "ai-forever/ru-en-RoSBERTa" #"cointegrated/LaBSE-en-ru ai-sage/Giga-Embeddings-instruct
dir_model = "./roberta-embeddings" 

model = None
try:
    model = SentenceTransformer(dir_model, trust_remote_code=True)
except Exception as e:
    model = snapshot_download(
        repo_id=model_name,
        local_dir=dir_model
    )
    model = SentenceTransformer(dir_model, trust_remote_code=True)

client = QdrantClient(host="localhost", port=6333)
size_model = model.get_sentence_embedding_dimension()
collection = "demo_collection"

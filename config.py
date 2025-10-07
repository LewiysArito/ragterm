import logging
import os

from typing import List, Tuple
from ollama import Client as OllamaClient
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv 

load_dotenv()

def initialize_logger() -> logging.Logger:
    LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, LOG_LEVEL_NAME, logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log', encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logger initialized with level: {LOG_LEVEL_NAME}")
    return logger 

logger = initialize_logger()

def initialize_embedding_model()->SentenceTransformer:
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "ai-forever/ru-en-RoSBERTa")
    EMBEDDING_DIR_MODEL = "./model-embeddings" 

    try:
        logger.info(f"Wait, load model from: {EMBEDDING_DIR_MODEL}")
        model = SentenceTransformer(EMBEDDING_DIR_MODEL, trust_remote_code=True)
        logger.info("Model loaded successfully from local directory")
        return model
    except Exception as e:
        logger.warning(f"Local model not found: {e}")
        logger.info(f"Downloading model from HuggingFace: {EMBEDDING_MODEL_NAME}")
        
        try:
            snapshot_download(
                repo_id=EMBEDDING_MODEL_NAME,
                local_dir=EMBEDDING_DIR_MODEL
            )
            model = SentenceTransformer(EMBEDDING_DIR_MODEL, trust_remote_code=True)
            logger.info("Model downloaded and loaded successfully from HuggingFace")
            return model
        except Exception as download_error:
            logger.error(f"Failed to download model: {download_error}")
            raise

def initialize_qdrant()->Tuple[QdrantClient, int, int, str]:
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COUNT_RESULT_SEARCH =  int(os.getenv("QDRANT_COUNT_RESULT_SEARCH", 12))
    QDRANT_COUNT_DOCUMENT_FOR_RAG = int(os.getenv("QDRANT_COUNT_RESULT_SEARCH", 3))
    QDRANT_USE_GUARD = os.getenv("QDRANT_USE_GUARD") in ["True", "1", "true"]
    
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, https=QDRANT_USE_GUARD)
        client.get_collections()
        logger.info(f"Successfully connected to Qdrant at \
                     {"https" if QDRANT_USE_GUARD else "http"}://{QDRANT_HOST}:{QDRANT_PORT}")
        logger.debug((
            f"Successfully connected to Qdrant at {QDRANT_COUNT_RESULT_SEARCH}"
            "{QDRANT_COUNT_DOCUMENT_FOR_RAG}"
        ))

        logger.info(f"Successfully connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        return (client, QDRANT_COUNT_RESULT_SEARCH, QDRANT_COUNT_DOCUMENT_FOR_RAG, QDRANT_USE_GUARD)
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise

def initialize_ollama()->Tuple[OllamaClient, str, float]:
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
    OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434)) 
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", 0.7))
    OLLAMA_PROTOCOL = "https" if os.getenv("OLLAMA_USE_GUARD") in ["True", "1", "true"] else "http" 
    
    #if your language is not english, create new template 
    try:
        with open("text_prompt.txt", encoding="utf-8") as text:
            OLLAMA_TEMPLATE_TEXT = text.read()
        
        if len(OLLAMA_TEMPLATE_TEXT.strip()) == 0:
            logger.error("text_prompt.txt is empty")
            raise

        if "{query}" not in OLLAMA_TEMPLATE_TEXT:
            logger.error("Variable 'query' not in ollama template")
            raise

        if "{sources}" not in OLLAMA_TEMPLATE_TEXT:
            logger.error("Variable 'sources' not in ollama template")
            raise
    except Exception as e:
        OLLAMA_TEMPLATE_TEXT =  (
            "Based on the sources provided below, give an accurate answer to the question."
            "\n\n"
            "QUESTION:"
            "{query}"
            "\n\n"
            "SOURCES:"
            "{sources}"
            "\n\n"
            "INSTRUCTIONS:"
            "- Answer as accurately as possible based on the sources"
            "- If the information is not in the sources, state this clearly"
            "- Do not mention the sources in your answer, just provide the information"
        )
        
    try:
        client = OllamaClient(f"{OLLAMA_PROTOCOL}://{OLLAMA_HOST}:{OLLAMA_PORT}")
        client.list()
        logger.info(f"Successfully connected to Ollama at {OLLAMA_HOST}:{OLLAMA_PORT}")
        logger.debug(f"Ollama model is '{OLLAMA_MODEL}' and temperature param equal {OLLAMA_TEMPERATURE}")
        return (client, OLLAMA_MODEL, OLLAMA_TEMPERATURE) 
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        raise

def initialize_chunk() -> Tuple[int, int, List[str]]:
    CHUNK_SIZE = os.getenv("CHUNK_SIZE", 800)
    CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP", 200)

    #Single letters that can be words found among the uploaded documents 
    CHUNK_NOT_EXTRACT_SYMBOL = list(map(lambda value: value.split(),
         os.getenv("CHUNK_NOT_EXTRACT_SYMBOL","a,o,i")).split(","))
    
    logger.debug(f"Chunk parameters - size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
    logger 
    return (CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_NOT_EXTRACT_SYMBOL)

def get_document_dir() -> str:
    DOCUMENT_DIR = "files"
    
    os.makedirs(DOCUMENT_DIR, exist_ok=True)
    logger.info(f"Document directory: {DOCUMENT_DIR}")
    return DOCUMENT_DIR

OLLAMA_CLIENT, OLLAMA_MODEL, OLLAMA_TEMPERATURE = initialize_ollama()
EMBEDDING_MODEL = initialize_embedding_model()
(
    QDRANT_CLIENT,
    QDRANT_COUNT_RESULT_SEARCH,
    QDRANT_COUNT_DOCUMENT_FOR_RAG,
    QDRANT_BASE_TEXT
) = initialize_qdrant()
CHUNK_SIZE, CHUNK_OVERLAP = initialize_chunk()
import re
import abc
import uuid

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from qdrant_client import models
from langchain_core.documents import Document

from config import (
    QDRANT_CLIENT, EMBEDDING_MODEL,
    QDRANT_LIMIT_RESULT_SEARCH, QDRANT_COUNT_DOCUMENT_FOR_RAG, CHUNK_NOT_EXTRACT_SYMBOLS, 
)

@dataclass(frozen=True)
class CustomPage():
    number_page: int
    page_content: str

@dataclass(frozen=True)
class CustomChunk():
    number_page: int
    page_content: str
    page_link: Optional[str]

class AbstractVectorDB(abc.ABC):
    """
    Abstract class for working with vector databases
    """

    @abc.abstractmethod
    def upload_chunks(self, chunks: List[Document], source: str) -> None:
        """
        Uploads chunks to the vector database
        """
        NotImplementedError

    @abc.abstractmethod
    def upload_pages(self, pages: List[Document], source: str) -> None:
        """
        Uploads pages to the vector database
        """
        NotImplementedError

    @abc.abstractmethod
    def search(self, query: str, limit: int, score_threshold: float) -> List[Dict[str, Any]]:
        """
        Searches in the vector database
        """
        NotImplementedError

    @abc.abstractmethod
    def get_all_collections(self):
        """
        Retrieves all collections from the vector database
        """
        NotImplementedError

    @abc.abstractmethod
    def get_chunk_by_page(
        self,
        page: int,
        collection: str
    )->Optional[str]:
        """
        Retrieves a chunk by page from the vector database
        """
        NotImplementedError

class QdrantRepository(AbstractVectorDB):
    """
    Implementation for working with Qdrant vector database
    """

    def __init__(self, 
            client = QDRANT_CLIENT, 
            model  = EMBEDDING_MODEL,
        ):
        self.client = client
        self.model = model
    
    def _clean_text(self, text: str, max_length: int = 2000,
        single_letter_words = CHUNK_NOT_EXTRACT_SYMBOLS
    ):
        """
        Function text cleaning
        """
        if not text or not isinstance(text, str):
            return ""
        
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
        
        words = text.split()
        clean_words = []
        
        for word in words:
            if (len(word) == 1 and 
                word in single_letter_words):
                clean_words.append(word)
            elif (len(word) == 1 and 
                  (word.isdigit() or word in ".,!?;:")):
                clean_words.append(word)
            elif len(word) > 1:
                clean_words.append(word)
            elif len(word) == 1 and not word.isalnum():
                continue
        
        clean_text = ' '.join(clean_words)
        
        if max_length and len(clean_text) > max_length:
            truncated = clean_text[:max_length]
            
            last_sentence_end = max(
                truncated.rfind('. '),
                truncated.rfind('! '),
                truncated.rfind('? ')
            )
            
            if last_sentence_end > max_length * 0.7:
                clean_text = truncated[:last_sentence_end + 1] + ".."
            else:
                clean_text = truncated + "..."
        
        return clean_text.strip()

    def _clean_russian_text(self, text: str) -> str:
        """
        Cleans Russian text by applying specific rules
        """
        return self._clean_text(text, single_letter_words=["а", "в", "к", "о", "с", "у", "и"])
    
    def _clean_english_text(self, text: str) -> str:
        """
        Cleans English text by applying specific rules
        """
        return self._clean_text(text)
    
    def _ensure_collection_exists(self, collection: str) -> None:
        """
        Creates a collection if it does not exist
        """
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                )
            )

    def upload_chunks(self, collection: str, chunks: List[CustomChunk], source: str) -> None:
    
        """
        Uploads chunks of text to the vector database
        """
        texts = []
        clean_chunks: List[CustomChunk] = []
        
        for chunk in chunks:
            clean_text = self._clean_text(chunk.page_content)
            if len(clean_text.strip()) > 50:
                texts.append(clean_text)
                clean_chunks.append(chunk)
        
        if not texts:
            return
        
        embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=32)
        self._ensure_collection_exists(collection=collection)
        
        points = []
        for i, (embedding, chunk, clean_text) in enumerate(zip(embeddings, clean_chunks, texts)):                
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "text": clean_text,
                    "number_page": int(chunk.number_page) + 1,
                    "source": source,
                    "chunk_index": i,
                    "page_link": chunk.page_link, #link
                    "type": "chunk"
                }
            ))
        
        self.client.upload_points(
            collection_name=collection,
            points=points
        )
    
    def upload_pages(self, collection: str, pages: List[CustomPage], source: str) -> None:
        """
        Uploads full pages to the vector database
        """
        texts = []
        clean_pages: List[CustomPage] = []
        
        for page in pages:
            clean_text = self._clean_text(page.page_content)
            if len(clean_text.strip()) > 50:
                texts.append(clean_text)
                clean_pages.append(page)
        
        if not texts:
            return
        
        embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=32)
        self._ensure_collection_exists(collection=collection)
        
        points = []
        for i, (embedding, page, clean_text) in enumerate(zip(embeddings, clean_pages, texts)):            
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "text": clean_text,
                    "number_page": int(page.number_page) + 1,
                    "source": source,
                    "page_index" : i,
                    "type": "page"
                }
            ))
        
        self.client.upload_points(
            collection_name=collection,
            points=points
        )
    
    def search(self, collection: str, query: str, limit = QDRANT_LIMIT_RESULT_SEARCH, score_threshold: float = 0.4) -> List[Dict[str, Any]]:
        """
        Searches in the vector database
        """
        query_embedding = self.model.encode([query]).tolist()[0]
        search_result = self.client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold
        )
        
        return [{
            "text": hit.payload["text"],
            "score": hit.score,
            "source": hit.payload.get("source", ""),
            "number_page": hit.payload.get("number_page", ""),
            "type": hit.payload.get("type", "chunk")
        } for hit in search_result]

    def get_relevant_documents(
        self,
        collection: str,
        query: str,
        max_pages = QDRANT_COUNT_DOCUMENT_FOR_RAG,
    )->List[int]:
        """Extract unique page numbers from search results for RAG context."""
        search_results = self.search(collection, query)
        pages = list([
            int(result["number_page"]) 
            for result in search_results
            if result.get("number_page")
        ])

        unique_pages = []
        current_index = 0
        while len(unique_pages) < max_pages and len(pages) > current_index:  
            if pages[current_index] not in unique_pages:
                unique_pages.append(pages[current_index])
            current_index += 1
        
        return unique_pages

    def get_all_collections(self) -> list[str]:
        """
        Retrieves all collections from the vector database
        """
        return [collection.name for collection in self.client.get_collections().collections]
    
    def delete_collection(self, collection_name: str)->bool:
        """
        Delete collection from vector database
        """
        return self.client.delete_collection(collection_name)

    def delete_collections(self, collections_name: str) -> List[str]:
        """
        Deletes multiple collections from the vector database
        """
        deleted_collections: List[str] = []
        for collection_name in collections_name:
            self.client.delete_collection(collection_name)
            deleted_collections.append(collection_name)

        return deleted_collections
    
    def get_chunk_by_page(
        self,
        collection: str,
        page: int,
    )->Optional[str]:
        """
        Retrieves a chunk (page) of a document
        """
        search_result = self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="number_page",
                        match=models.MatchValue(value=page)
                )]
            ),
            limit=1,
        )

        if not search_result:
            return None
        
        return search_result[0][0].payload["text"]

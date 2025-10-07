from dataclasses import dataclass
from qdrant_client import QdrantClient, models
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import re
import abc
from typing import List, Dict, Any, Optional, Union
import uuid

@dataclass(frozen=True)
class CustomPage(unsafe_hash=True):
    number_page: int
    page_content: str
    page_link: Optional[str]

@dataclass(frozen=True)
class CustromChunk(unsafe_hash=True):
    number_page: int
    page_content: str


class AbstractVectorDB(abc.ABC):
    """Абстрактный класс для работы с векторными базами данных"""
    
    @abc.abstractmethod
    def upload_chunks(self, chunks: List[Document], source: str) -> None:
        """Загружает чанки в векторную БД"""
        pass
    
    @abc.abstractmethod
    def upload_pages(self, pages: List[Document], source: str) -> None:
        """Загружает страницы в векторную БД"""
        pass
    
    @abc.abstractmethod
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Поиск в векторной БД"""
        pass

class QdrantRepository(AbstractVectorDB):
    """Реализация для работы с Qdrant"""
    
    def __init__(self, 
            client: QdrantClient, 
            model: SentenceTransformer,
            chunk_size: int = 800,
            chunk_overlap: int = 150):
        self.client = client
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def _clean_russian_text(self, text: str, max_length: int = 2000) -> str:
        """Улучшенная очистка русского текста"""
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
                word in ["а", "в", "к", "о", "с", "у", "и"]):
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
    
    def _ensure_collection_exists(self, collection: str) -> None:
        """Создает коллекцию если она не существует"""
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                )
            )
    
    def upload_chunks(self, collection: str, chunks: List[CustromChunk], source: str) -> None:
        """Загружает чанки в векторную БД"""
        texts = []
        clean_chunks: List[CustromChunk] = []
        
        for chunk in chunks:
            clean_text = self._clean_russian_text(chunk.page_content)
            if len(clean_text.strip()) > 50:
                texts.append(clean_text)
                clean_chunks.append(chunk)
        
        if not texts:
            return
        
        embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=32)
        self._ensure_collection_exists()
        
        points = []
        for i, (embedding, chunk, clean_text) in enumerate(zip(embeddings, clean_chunks, texts)):                
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "text": clean_text,
                    "number_page": chunk.number_page + 1,
                    "source": source,
                    "chunk_index": i,
                    "page_link": chunk.page_link, #ссылка на коллекцию где страницы целиком находятся
                    "type": "chunk"
                }
            ))
        
        self.client.upload_points(
            collection_name=collection,
            points=points
        )
    
    def upload_pages(self, collection: str, pages: List[CustomPage], source: str) -> None:
        """Загружает полные страницы в векторную БД"""
        texts = []
        clean_pages: List[CustomPage] = []
        
        for page in pages:
            clean_text = self._clean_russian_text(page.page_content)
            if len(clean_text.strip()) > 50:
                texts.append(clean_text)
                clean_pages.append(page)
        
        if not texts:
            return
        
        embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=32)
        self._ensure_collection_exists()
        
        points = []
        for i, (embedding, page, clean_text) in enumerate(zip(embeddings, clean_pages, texts)):            
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "text": clean_text,
                    "number_page": page.page_content + 1,
                    "source": source,
                    "page_index" : i,
                    "type": "page"
                }
            ))
        
        self.client.upload_points(
            collection_name=collection,
            points=points
        )
    
    def search(self, collection: str, query: str, limit: int = 5, score_threshold: float = 0.4) -> List[Dict[str, Any]]:
        """Поиск в векторной БД"""
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
            "number_page": hit.payload.get("page", ""),
            "type": hit.payload.get("type", "chunk")
        } for hit in search_result]

    def get_chunk_by_page(
        self,
        page: int,
        collection: str
    ):
        """Получает определенную страницу документа"""
        
        
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

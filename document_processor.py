import abc
from typing import Any, Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFPlumberLoader

class AbstractDocumentProcessor(abc.ABC):
    """Абстрактный класс для обработки документов"""
    
    @abc.abstractmethod
    def load_and_split_file(self, chunks: List[Document], source: str) -> None:
        """Загружает документ и возвращает страницы и чанки"""
        pass

class PDFProcessor:
    """Класс для обработки PDF файлов"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    def load_and_split_file(self, file_path: str) -> Dict[str, List[Document]]:
        """Загружает документ и возвращает страницы и чанки"""
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        
        pages = self._extract_full_pages(documents)
        
        chunks = self.text_splitter.split_documents(documents)
        
        return {
            "pages": pages,
            "chunks": chunks
        }
    
    def _extract_full_pages(self, documents: List[Document]) -> List[Document]:
        """Извлекает полный текст каждой страницы"""
        page_texts = {}
        
        for doc in documents:
            page_num = doc.metadata.get("page", 0)
            if page_num not in page_texts:
                page_texts[page_num] = []
            page_texts[page_num].append(doc.page_content)
        
        full_pages = []
        for page_num, texts in page_texts.items():
            full_page_text = " ".join(texts)
            full_pages.append(Document(
                page_content=full_page_text,
                metadata={"page": page_num}
            ))
        
        return full_pages

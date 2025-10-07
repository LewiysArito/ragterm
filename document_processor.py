import abc
from typing import Any, Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFPlumberLoader

from config import CHUNK_SIZE, CHUNK_OVERLAP

class AbstractDocumentProcessor(abc.ABC):
    """Abstract class for handling documents"""
    
    @abc.abstractmethod
    def load_and_split_file(self, chunks: List[Document], source: str) -> None:
        """Loads document and returns page content and chunks"""
        pass

class PDFProcessor:
    """Class for handling pdf files"""
    
    def __init__(self, chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    def load_and_split_file(self, file_path: str) -> Dict[str, List[Document]]:
        """Loads document and returns page content and chunks"""
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        
        pages = self._extract_full_pages(documents)
        
        chunks = self.text_splitter.split_documents(documents)
        
        return {
            "pages": pages,
            "chunks": chunks
        }
    
    def _extract_full_pages(self, documents: List[Document]) -> List[Document]:
        """Extracts full text for every page"""

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

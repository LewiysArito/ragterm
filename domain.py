import os
from typing import List

from ollama_processor import OllamaProcessor
from qdrant_repository import CustomPage, QdrantRepository, CustomChunk
from config import get_document_dir
from document_processor import PDFProcessor

qdrant_rep = QdrantRepository()
pdf_processor = PDFProcessor()
ollama_processor = OllamaProcessor()   
document_dir = get_document_dir()

class DocumentVector:
    """
    Class for managing document vectors in a vector database
    """
    def __init__(self, 
            qdrant_rep = qdrant_rep,
            pdf_processor = pdf_processor,
            ollama_processor = ollama_processor
        ):
        self.qdrant_rep =  qdrant_rep
        self.pdf_processor = pdf_processor
        self.ollama_processor = ollama_processor
        self.filedir = os.path.join(os.getcwd(), )

    def _get_collection_basename(filename: str):
        return filename.split(".")[0]

    def _get_chunks_collection_name(basename: str):
        return basename + "_chunks"
    
    def _get_pages_collection_name(basename: str):
        return basename + "_pages"

    def upload_file(self, filename: str):
        """
        Upload a file to the vector database.

        Args:
            file_name (str): The name of the file to upload.
        """

        basename = self._get_collection_basename(filename)
        col_pages_name = self._get_pages_collection_name(basename)
        col_chunks_name = self._get_chunks_collection_name(basename)
        
        documents = pdf_processor.load_and_split_file(
            filename
        )

        pages = [CustomPage(document_page.metadata.get("page", 0), document_page.page_content) 
            for document_page in documents["pages"]
        ]

        chunks = [CustomChunk(document_page.metadata.get("page", 0), document_page.page_content, col_pages_name) 
            for document_page in documents["chunks"]
        ]

        qdrant_rep.upload_pages(
            collection=col_pages_name,
            pages=pages,
            source=filename
        )

        qdrant_rep.upload_chunks(
            collection=col_chunks_name,
            chunks=chunks,
            source=filename
        )

    def search_from_file(self, filename: str, prompt: str):
        """
        Search 
        """

        basename = self._get_collection_basename(filename)
        col_chunks_name = self._get_chunks_collection_name(basename)

        values = qdrant_rep.search(col_chunks_name, prompt)
        
        return values

    def delete_file(self, filename: str):
        """
        Удаляет файл и всю информацию из векторной базы данных
        """
        
        basename = self._get_collection_basename(filename)
        col_pages_name = self._get_pages_collection_name(basename)
        col_chunks_name = self._get_chunks_collection_name(basename)

    def show_all_files(self, filename: str):
        """
        показывает
        """
        pass

    def show_all_collections(self)->List[str]:
        """Show All"""
        return self.qdrant_rep.get_all_collections()
    

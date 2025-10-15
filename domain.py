import os
import shutil

from typing import List, Optional
from ollama_processor import OllamaProcessor, AbstractLLMClient
from qdrant_repository import CustomPage, QdrantRepository, CustomChunk, AbstractVectorDB
from config import get_document_dir
from document_processor import PDFProcessor, AbstractDocumentProcessor

DOCUMENT_DIR = get_document_dir()

class DocumentVector:
    """
    Class for managing document vectors in a vector database
    """
    CHUNK_COLLECTION_ENDING = "_chunks" 
    PAGE_COLLECTION_ENDING = "_pages"

    def __init__(self, 
            vector_rep: Optional[AbstractVectorDB] = None,
            pdf_processor: Optional[AbstractDocumentProcessor] = None,
            llm_processor: Optional[AbstractLLMClient] = None,
            document_dir = DOCUMENT_DIR
        ):

        if not vector_rep:
            vector_rep = QdrantRepository()
        
        if not pdf_processor:
            pdf_processor = PDFProcessor()
        
        if not llm_processor:
            llm_processor = OllamaProcessor()
        
        self.vector_rep = vector_rep
        self.pdf_processor = pdf_processor
        self.llm_processor = llm_processor

        self.filedir = os.path.join(os.getcwd(), document_dir)
                
    def _get_collection_basename(self, filename: str):
        """
        Transforms filename to basename
        """
        return filename.split(".")[0]

    def _get_chunks_collection_name(self, basename: str):
        """
        Transforms basename to collection chunks
        """
        return basename + self.CHUNK_COLLECTION_ENDING
    
    def _get_pages_collection_name(self, basename: str):
        """Transforms basename to collection pages"""
        return basename + self.PAGE_COLLECTION_ENDING

    def upload_file(self, file_path: str):
        """
        Upload a file to the vector database.

        Args:
            file_path (str): Path from which file will be downloaded
        """

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        
        filename = os.path.basename(file_path)
        basename = self._get_collection_basename(filename)
        col_pages_name = self._get_pages_collection_name(basename)
        col_chunks_name = self._get_chunks_collection_name(basename)
        
        new_path = shutil.copy2(file_path, os.path.join(self.filedir, filename))

        documents = self.pdf_processor.load_and_split_file(
            new_path
        )

        pages = [CustomPage(document_page.metadata.get("page", 0), document_page.page_content) 
            for document_page in documents["pages"]
        ]

        chunks = [CustomChunk(document_chunk.metadata.get("page", 0), document_chunk.page_content, col_pages_name) 
            for document_chunk in documents["chunks"]
        ]

        self.vector_rep.upload_pages(
            collection=col_pages_name,
            pages=pages,
            source=filename
        )

        self.vector_rep.upload_chunks(
            collection=col_chunks_name,
            chunks=chunks,
            source=filename
        )

    def find_chunks_from_file(self, filename: str, query: str)->List[str]:
        """
        Search for relevant text chunks in the specified file.
        """

        basename = self._get_collection_basename(filename)
        col_chunks_name = self._get_chunks_collection_name(basename)

        values = self.vector_rep.search(col_chunks_name, query) 
        return [value["text"] for value in values]
    
    def rag_search_from_file(self, filename: str, query: str)->str:
        """
        Searches for relevant pages in the file and also generates a template, which is sent to LLM for processing with the expectation of the result
        """
        basename = self._get_collection_basename(filename)
        col_chunks_name = self._get_chunks_collection_name(basename)
        col_pages_name = self._get_pages_collection_name(basename)

        pages = self.vector_rep.get_relevant_documents(col_chunks_name, query)

        texts = []
        for page in pages:
            text = self.vector_rep.get_chunk_by_page(col_pages_name, page)
            if text:
                texts.append(text)

        result = self.llm_processor.generate_prompt_from_template(
            query,
            "\n\n".join(texts),
        )

        return result

    def delete_rag_file(self, filename: str):
        """
        Delete collection and file by filename
        """
        file_path = os.path.join(self.filedir, filename)  
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {filename} not found")
        
        basename = self._get_collection_basename(filename)
        col_pages_name = self._get_pages_collection_name(basename)
        col_chunks_name = self._get_chunks_collection_name(basename)

        self.vector_rep.delete_collection(
            collection_name=col_chunks_name, 
        )

        self.vector_rep.delete_collection(
            collection_name=col_pages_name, 
        )

        os.remove(file_path)

    def clear_all(self)->list[str]:
        """
        Deleted from vector database all documents, loaded for rag
        """
        collections = list(filter(
            lambda value:
            value.endswith(self.CHUNK_COLLECTION_ENDING) or 
            value.endswith(self.PAGE_COLLECTION_ENDING),
            self.vector_rep.get_all_collections()
        ))

        deleted_collections = self.vector_rep.delete_collections(collections)

        for filename in os.listdir(self.filedir):
            file_path = os.path.join(self.filedir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        return deleted_collections


    def show_all_collections(self)->List[str]:
        """Show all collections for rag"""

        return list(filter(
            lambda value:
            value.endswith(self.CHUNK_COLLECTION_ENDING) or 
            value.endswith(self.PAGE_COLLECTION_ENDING),
            self.vector_rep.get_all_collections()
        ))
    
    
    

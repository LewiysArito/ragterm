from qdrant_repository import CustomPage, QdrantRepository, CustomChunk
from config import client, model, size_model
from document_processor import PDFProcessor

qdrant_rep = QdrantRepository(
    client=client,
    model=model
)

file_name = "Копия HTML.pdf"
collection_base_name = file_name.split(".")[0]
collection_pages_name = collection_base_name + "_pages"
collection_chunks_name = collection_base_name + "_chunks"

pdf_processor = PDFProcessor()
documents = pdf_processor.load_and_split_file(
    file_name
)

pages = [CustomPage(document_page.metadata.get("page", 0), document_page.page_content) 
    for document_page in documents["pages"]
]

chunks = [CustomChunk(document_page.metadata.get("page", 0), document_page.page_content, collection_pages_name) 
    for document_page in documents["chunks"]
]

qdrant_rep.upload_pages(
    collection=collection_pages_name,
    pages=pages,
    source=file_name
)

qdrant_rep.upload_chunks(
    collection=collection_chunks_name,
    chunks=chunks,
    source=file_name
)

values = qdrant_rep.search(collection_chunks_name, "Как разместить свой сайт в интернете?", limit=20, score_threshold=0.4)

print(*[value["text"] + " "+ str(value["score"]) + " " + str(value["number_page"]) + "\n\n" for value in values])
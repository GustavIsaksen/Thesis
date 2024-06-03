from langchain.storage import LocalFileStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings

from document_loader import get_file_content, get_files
from splitters import chunk_file
from file_definition import File

import time

def embed_file_local(files: File):
    """Embeds the files using the OpenAIEmbeddings and stores the embeddings in the file class under vector_store field."""

    underlying_embeddings = OpenAIEmbeddings()

    store = LocalFileStore("./cache/")

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, store, namespace=underlying_embeddings.model
    )

    for file in files:
        start_time = time.time()
        file.vector_store = FAISS.from_documents(file.split_content, cached_embedder)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")


if __name__ == "__main__":
    
    # print(list(store.yield_keys()))

    word_files = get_files(file_type='word',parent_folder_list=['2023','2024'],num_documents=1)
    # pdf_files = get_files(file_type='pdf',parent_folder_list=['other'],num_documents=2)
    # files = word_files + pdf_files
    files = word_files 

    get_file_content(files) 
    chunk_file(files)
    embed_file_local(files)

    



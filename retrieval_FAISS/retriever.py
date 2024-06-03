from langchain.retrievers import ParentDocumentRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List

from document_loader import get_all_files, get_file_content, get_files
from splitters import chunk_file
from embedder import embed_file_local
from file_definition import File

from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

class CustomRetriever(BaseRetriever):
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return [Document(page_content=query)]

def retrieve_vector_store_backed(question, files : list[File]):
    """Retrieve documents using a vector store backed retriever. You need to run each FAISS db against the retriever. """

    output_file_name = f"faiss_output/retriever_output {question[0:50]}.txt"

    with open(output_file_name, "w") as output_file:
        output_file.write(f"Question: \n{question}\n\n")
        relevant_docs = 0
        num_files = 0

        retriever_output_string = ""

        for file in files:
            num_files += 1
            retriever = file.vector_store.as_retriever(
                search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7}
                # search_kwargs={"k": 1}
                # search_type="mmr"
            )
            docs = retriever.get_relevant_documents(question)
            if len(docs) == 0: # if no documents are found, go to next file
                continue
            filename_string = f"File: {file.file_name}\n"
            retriever_output_string = retriever_output_string + filename_string

            # output_file.write(f"File: {file.file_name}\n")
            for doc in docs:
                relevant_docs += 1
                doc_string = doc.page_content + "\n\n" 
                retriever_output_string = retriever_output_string + doc_string
                

        relevant_files_string = f"Looked through {num_files} relevant files \n\n"
        relevant_docs_string = f"Found {relevant_docs} relevant paragraphs \n\n"
        print(relevant_files_string)
        print(relevant_docs_string)

        output_file.write(relevant_files_string)
        output_file.write(relevant_docs_string)
        output_file.write(retriever_output_string)
        print(f"wrote relevant paragraphs to {output_file_name}")


def retrieve_parent_document(files : list[File]):
    """Retrieve parent-document / larger chunk window using a vector store backed retriever. You need to run each FAISS db against the retriever. """

    #TODO: NOT WORKING because of chromadb
    print("Not implemented yet")
    return

    #assumes that the files have already been split 
    docs = []
    for file in files:
        docs.extend(file.split_content)

    # This text splitter is used to create the child documents
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
    collection_name="full_documents", embedding_function=OpenAIEmbeddings()
    )
    # The storage layer for the parent documents
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    )

    print(docs)

    retriever.add_documents(docs, ids=None)

    print(list(store.yield_keys()))


if __name__ == "__main__":

    # medium data-set     
    files = get_files('word',['2023','2024'],100)
    word_files = get_files(file_type='word',parent_folder_list=['2023','2024'],num_documents=15) 
    esg_report_pdf_files = get_files(file_type='pdf',parent_folder_list=['other'],num_documents=10)
    files = word_files + esg_report_pdf_files

    # files = get_all_files(2) # TODO: error on high number of files
    
    get_file_content(files) 
    chunk_file(files)
    embed_file_local(files)

    question = input("Which keyword / question would you like to search for? \n")    
    # retrieve_parent_document(files) #not working
    retrieve_vector_store_backed(question, files)

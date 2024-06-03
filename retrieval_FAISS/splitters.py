from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import get_file_content, get_files, get_all_files
from file_definition import File

def recursive_char_splitter(text: str) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.create_documents([text])
    return texts



def chunk_file(files : list[File]):
    for file in files:
        if file.file_type == 'word':
            file.text_content = file.document_to_text()
            file.split_content = recursive_char_splitter(file.text_content)
        elif file.file_type == 'pdf':
            file.split_content = recursive_char_splitter(file.text_content)
        else:
            print("cant handle {file.file_type} file type") #TODO: Implement this for other file types

if __name__ == "__main__":
    # print("testing word files: ")
    # files = get_files('word',['2023'],1)
    # get_file_content(files) 
    # chunk_file(files)
    # print(files[0].split_content)

    # print("testing pdf files: ")
    # files = get_files('pdf',['other'],1)
    # get_file_content(files) 
    # chunk_file(files)
    # print(files[0].split_content)


    # word_files = get_files(file_type='word',parent_folder_list=['2023','2024'],num_documents=2)
    # pdf_files = get_files(file_type='pdf',parent_folder_list=['other'],num_documents=2)
    # files = word_files + pdf_files

    files = get_all_files(10)
    get_file_content(files) 
    chunk_file(files)

    for file in files:
        print(file.split_content)



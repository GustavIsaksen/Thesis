from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import get_file_content, get_files, get_all_files
from classes import File, Chunk

def recursive_char_splitter(text: str,chunk_size:int = 1000, chunk_overlap:int = 100) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.create_documents([text])
    return texts


def chunk_file(files : list[File]):
    for file in files:
        if file.file_type == 'word':
            file.text_content = file.document_to_text()
            split_content_list = recursive_char_splitter(file.text_content)

            for split_text in split_content_list:
                chunk = Chunk(text=split_text.page_content)
                file.chunks.append(chunk)
            

        elif file.file_type == 'pdf':
            split_content_list = recursive_char_splitter(file.text_content)

            for split_text in split_content_list:
                chunk = Chunk(text=split_text.page_content)
                file.chunks.append(chunk)
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

    # files = get_all_files(10)
    
    try:
        files = get_files(file_type='word',parent_folder_list=['2024'],num_documents=1)

        get_file_content(files) 
        chunk_file(files)
    except Exception as e:
        print(f"failed word splitting: {e}")


    try:
        files = get_files(file_type='pdf',parent_folder_list=['other'],num_documents=1)

        get_file_content(files) 
        chunk_file(files)
    except Exception as e:
        print(f"failed pdf splitting: {e}")

    # for file in files:
    #     print(file.split_content)



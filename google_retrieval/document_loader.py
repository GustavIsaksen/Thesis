import io
from docx import Document
from import_file_from_GoogleDrive import GoogleDriveImporter
from classes import File
import PyPDF2

import yaml
from tqdm import tqdm

# Function to print IDs and companies
def get_file_ids(section1,section2, file_overview : str):
    with open(file_overview, 'r') as file: 
        file_ids = yaml.safe_load(file)
    
    ids = []  # Initialize an empty list to hold the IDs
    for item in file_ids[section1][section2]:
        id = item.get('id')  # Using .get() for safer access
        if id not in ids:  # Check if the ID is already in the list to avoid duplicates
            ids.append(id)
    return ids

def get_files(file_type : str, parent_folder_list : list[str], num_documents : int, file_overview : str) -> list[File]:
    """"Loads the file ids from the organized_files.yaml file and returns a list of files.
        Set num_documents really high to get all docs
    """
    with open(file_overview, 'r') as file: 
        file_ids = yaml.safe_load(file)
    
    # Initialize a list of files
    files = []
    files_retrieved = 0
    for parent_folder in parent_folder_list:
        try:
            for item in file_ids[file_type][parent_folder]:
        
                id = item.get('id')
                file_name = item.get('file_name')
                full_path = item.get('full_path')
                file_type = file_type
                parent_folder = parent_folder

                file = File(id=id, file_name=file_name, full_path=full_path, file_type=file_type, parent_folder=parent_folder)
                files.append(file)
                files_retrieved += 1 
                if files_retrieved >= num_documents:
                    break
        except:
            print(f"Could not find any files in {parent_folder} of type {file_type}")

    return files

def get_all_files(num_documents: int, start_range : int, end_range : int, file_overview: str) -> list[File]:

    parent_folder_list = [str(year) for year in range(start_range, end_range)]

    word_files = get_files(file_type='word', parent_folder_list=parent_folder_list, num_documents=num_documents, file_overview=file_overview)
    print(f"found {len(word_files)} word files")
    for file in word_files:
        print(file.file_name)
    pdf_files = get_files(file_type='pdf', parent_folder_list=parent_folder_list, num_documents=num_documents, file_overview=file_overview)
    print(f"found {len(pdf_files)} pdf files")

    return word_files + pdf_files



def get_file_content(files : list[File]):
    """
    Populates the files with the content of the documents from google drive
    Returns: a list of files with the content.
    """
    driveSession = GoogleDriveImporter()

    for file in tqdm(files, desc="Downloading files from Google Drive"):
        file_content = driveSession.download_file(real_file_id=file.id)

        if file.file_type == 'word':
            document = Document(file_content)
            file.content = document
            file.text_content = file.document_to_text()

        elif file.file_type == 'pdf':
            reader = PyPDF2.PdfReader(file_content)
            file.text_content = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                file.text_content += page.extract_text() + "\n"
                # TODO: Add page number retrieval here?

if __name__ == "__main__":

    # word_files = get_files(file_type='word',parent_folder_list=['2023','2024'],num_documents=2)
    # pdf_files = get_files(file_type='pdf',parent_folder_list=['other'],num_documents=2)
    # files = word_files + pdf_files

    files = get_all_files(100, 2022, 2024, "2603_organized_files.yaml")

    get_file_content(files) 

    for file in files:
        print(file.text_content)
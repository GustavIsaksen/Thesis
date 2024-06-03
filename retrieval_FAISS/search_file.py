import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time
import random

import yaml

def exponential_backoff(retries):
    """Calculate sleep time in seconds before the next retry."""
    return min((2 ** retries) + random.uniform(0, 1), 60)  # Cap at 60 seconds

def get_full_folder_path(service, file_id, path=[]):
    """Recursively build the full folder path for the given file_id."""
    try:
        file = service.files().get(fileId=file_id, fields="parents, name").execute()
        name = file.get('name')
        parents = file.get('parents')
        if parents:
            parent_id = parents[0]  # Assuming single parent scenario
            path.insert(0, name)  # Prepend name to path
            return get_full_folder_path(service, parent_id, path)
        else:
            return '/'.join(path)  # Return the full path as string
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None

def safe_api_request(callable, *args, **kwargs):
    """Wrap Google Drive API calls with exponential backoff."""
    retries = 0
    while True:
        try:
            # Make sure to call .execute() on the callable to execute the API request
            result = callable(*args, **kwargs).execute()
            return result
        except HttpError as error:
            if error.resp.status in [403, 500, 503]:
                sleep_time = exponential_backoff(retries)
                print(f"Request failed with status {error.resp.status}, retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                retries += 1
                if retries >= 5:
                    print("Maximum retries reached. Aborting.")
                    raise
            else:
                raise


def organize_files(files,start_range: int, end_range : int, file_overview : str):
    """Organize files into a dictionary based on their folder structure.
    Start and end range are the years to consider for the relevant folders.
    file_overview is the path to the output file."""
    # Expanded to include both Word documents and PDFs
    organized_files = {"word": {}, "pdf": {}}
    relevant_folders = [str(year) for year in range(start_range, end_range)]

    for file in files:
        path_parts = file['full_path'].split('/')
        file_extension = file["file_name"].split('.')[-1].lower()
        file_type = "pdf" if file_extension == "pdf" else "word"  # Assuming only .docx and .pdf files are processed

        # Check if the path starts with "Documents/asdf"
        if len(path_parts) > 2 and path_parts[0] == "Documents" and path_parts[1] == "asdf":
            # Use the third layer (index 2) for folder name if the initial path is as expected
            folder_name = path_parts[2]
        elif len(path_parts) > 2 and path_parts[0] == "Documents" and path_parts[1] == "asdf" :
            folder_name = path_parts[2] # AKA ESG aasdf
        else:
            folder_name = "other"

        category = folder_name if folder_name in relevant_folders else "other"
        
        # Use file_type to select the right category (word or pdf)
        if category not in organized_files[file_type]:
            organized_files[file_type][category] = []
        organized_files[file_type][category].append({
            "id": file["id"],
            "file_name": file["file_name"],
            "full_path": file["full_path"]
        })

    with open(file_overview, 'w') as yamlfile:
        yaml.safe_dump(organized_files, yamlfile, default_flow_style=False)



def search_file(start_range: int, end_range : int, file_overview : str):
    creds, _ = google.auth.default()

    try:
        service = build("drive", "v3", credentials=creds)
        page_token = None
        file_overview = []
        while True:
            response = safe_api_request(
                service.files().list,
                q="(mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document' or mimeType='application/pdf')",
                spaces="drive",
                fields="nextPageToken, files(id, name, parents)",
                pageToken=page_token,
            )
            for file in response.get("files", []):
                full_path = get_full_folder_path(service, file.get("id"), [file.get("name")])
                print(f'Found file: {file.get("name")}, ID: {file.get("id")}, Full Path: {full_path}')
                file_overview.append({
                    "id": file.get("id"),
                    "file_name": file.get("name"),
                    "full_path": full_path
                })

            page_token = response.get("nextPageToken", None)
            if page_token is None:
                break
        organize_files(file_overview,start_range, end_range, file_overview)  # Call organize_files here, after collecting all files
    except HttpError as error:
        print(f"An error occurred: {error}")

if __name__ == "__main__":
    search_file(start_range = 2022, end_range = 2024, file_overview = "asdf.yaml")

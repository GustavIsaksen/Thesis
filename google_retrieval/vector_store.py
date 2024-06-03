from datetime import datetime
import json
import time
import subprocess
from google.cloud import aiplatform
from typing import Optional
from tqdm import tqdm
import yaml

from embedding import get_embeddings_wrapper


def upload_to_bucket(bucket_uri : str, file_path : str):
    # Upload the embeddings to a GCS bucket
    print(f"Uploading {file_path} to {bucket_uri}")
    subprocess.run(["gsutil", "cp", file_path, bucket_uri]) # you might need to do gcloud auth login first

def load_constants():
    with open('/workspaces/ESGenie/google_retrieval/google_config.yaml') as f:
        config = yaml.safe_load(f)

    # Extract required fields from config
    PROJECT_ID = config["PROJECT_ID"]
    LOCATION = config["LOCATION"]
    BUCKET_URI = config["BUCKET_URI"]
    DEPLOYED_INDEX_EXCEL_ID = config["DEPLOYED_INDEX_EXCEL_ID"]
    INDEX_ID = config["INDEX_ID"]
    ENDPOINT_ID = config["ENDPOINT_ID"]
    DEPLOYED_INDEX_WORD_AND_PDF_ID = config["DEPLOYED_INDEX_WORD_AND_PDF_ID"]

    return PROJECT_ID, LOCATION, BUCKET_URI, DEPLOYED_INDEX_EXCEL_ID, INDEX_ID, ENDPOINT_ID, DEPLOYED_INDEX_WORD_AND_PDF_ID


if __name__ == "__main__":
    PROJECT_ID, LOCATION, BUCKET_URI, DEPLOYED_INDEX_EXCEL_ID, INDEX_ID, ENDPOINT_ID, DEPLOYED_INDEX_WORD_AND_PDF_ID = load_constants()

    # my_index_id = "[your-index-id]"  # @param {type:"string"}
    # my_index = aiplatform.MatchingEngineIndex(my_index_id)    

    # my_index_endpoint_id = "[your-index-endpoint-id]"  # @param {type:"string"}
    # my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(my_index_endpoint_id)

    ## test_data
    # embeddings_file = "/workspaces/ESGenie/questions_with_embeddings.json"
    # bucket_location = BUCKET_URI + "/excel_test_set_embedded_questions"
    
    # training data
    embeddings_file = "/workspaces/ESGenie/json_files/1104_QnA_embedded.json"
    bucket_location = BUCKET_URI + "/excel_train_set_embedded_questions"
    upload_to_bucket(bucket_location, embeddings_file)


    # word files
    # embeddings_file = "/workspaces/ESGenie/pdf_files_with_embeddings.json"
    # bucket_location = BUCKET_URI + "/word_and_pdf_train_set_embedded"
    # upload_to_bucket(bucket_location, embeddings_file)

    
import json
import pandas as pd
import numpy as np

from google.cloud import aiplatform

from vector_store import load_constants
from embedding import get_embeddings_wrapper

NUM_NEIGHBORS_WORD = 5
NUM_NEIGHBORS_EXCEL = 5
NUM_RESPONSES_OUTPUT = 5
MAX_OUTPUT_CHAR = 5000
EXCEL_QNA = '/asdf.json'
WORD_AND_PDF_FILE = '/asdf.json'

import logging
import os



# Configure logging to include the script name in each message
logging.basicConfig(filename='application.log', level=logging.INFO, 
                    format=f'%(asctime)s - %(levelname)s - %(message)s')



def load_df_from_jsonl(filename) -> pd.DataFrame:
    # read json file
    with open(filename, 'r') as f:
        df = pd.read_json(filename, lines=True)
        # data = f.readlines()
    return df


def load_df(filename) -> pd.DataFrame:
    # read json file
    with open(filename, 'r') as f:
        data = json.load(f)

    # add id 
    for i, d in enumerate(data):
        d['id'] = i

    df = pd.DataFrame(data)
    return df


def match_id_to_df(df: pd.DataFrame, response: dict) -> pd.DataFrame:
    df_output = pd.DataFrame(columns=["Similarity", "Question", "Answer", "Response","Topic", "Sheet name", "file name"])

    for idx, neighbor in enumerate(response[0]):
        id = np.int64(neighbor.id)  # Adjusted for dictionary access
        similar = df.query("id == @id", engine="python")
        if not similar.empty:
            new_row = {
                "Similarity": round(neighbor.distance * 100, 2),
                "Question": similar.Question.values[0] if similar.Question.values.size > 0 else np.nan,
                "Answer": similar.Answer.values[0] if similar.Answer.values.size > 0 else np.nan,
                "Topic": similar.Topic.values[0] if similar.Topic.values.size > 0 else np.nan,
                "Sheet name": similar["Sheet name"].values[0] if similar["Sheet name"].values.size > 0 else np.nan,
                "file name": similar["file name"].values[0] if similar["file name"].values.size > 0 else np.nan
            }

            # concat question and answer
            new_row["Response"] = f"- Question: {new_row['Question']}\nAnswer: {new_row['Answer']}"

            df_output.loc[idx] = new_row

    return df_output

def match_id_to_word_and_pdf_df(df: pd.DataFrame, response: dict) -> pd.DataFrame:
    df_output = pd.DataFrame(columns=["Similarity", "Response", "file name"])

    for idx, neighbor in enumerate(response[0]):
        id = np.int64(neighbor.id)  # Adjusted for dictionary access
        similar = df.query("id == @id", engine="python")
        if not similar.empty:
            new_row = {
                "Similarity": round(neighbor.distance * 100, 2),
                "Response": similar.chunk_text.values[0] if similar.chunk_text.values.size > 0 else np.nan,
                "file name": similar["file_name"].values[0] if similar["file_name"].values.size > 0 else np.nan
            }
            df_output.loc[idx] = new_row
        else:
            # print(f"Couldn't match id: {id} to the dataframe")
            logging.info(f"Couldn't match id: {id} to the dataframe")

    return df_output


def query_index_excel(query: str) -> pd.DataFrame:
    PROJECT_ID, LOCATION, BUCKET_URI, DEPLOYED_INDEX_EXCEL_ID, INDEX_ID, ENDPOINT_ID, DEPLOYED_INDEX_WORD_AND_PDF_ID = load_constants()
    

    # Initialize AI Platform
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    my_index = aiplatform.MatchingEngineIndex(INDEX_ID) # TODO: Delete
    my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(ENDPOINT_ID)

    query_embedding = get_embeddings_wrapper([query])

    response = my_index_endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_EXCEL_ID,
        queries=query_embedding,
        num_neighbors=NUM_NEIGHBORS_EXCEL,
    )
    response_df_excel = match_id_to_df(load_df_from_jsonl(EXCEL_QNA), response)

    return response_df_excel


def query_index_word_and_pdf(query: str) -> pd.DataFrame:
    PROJECT_ID, LOCATION, BUCKET_URI, DEPLOYED_INDEX_EXCEL_ID, INDEX_ID, ENDPOINT_ID, DEPLOYED_INDEX_WORD_AND_PDF_ID = load_constants()
    
    # Initialize AI Platform
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    my_index = aiplatform.MatchingEngineIndex(INDEX_ID) # TODO: Delete
    my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(ENDPOINT_ID)

    query_embedding = get_embeddings_wrapper([query])

    response = my_index_endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_WORD_AND_PDF_ID,
        queries=query_embedding,
        num_neighbors=NUM_NEIGHBORS_WORD,
    )
    response_df_word_and_pdf = match_id_to_word_and_pdf_df(load_df_from_jsonl(WORD_AND_PDF_FILE), response)

    return response_df_word_and_pdf


def select_highest_similarity_max_responses(response_df_word: pd.DataFrame,response_df_excel: pd.DataFrame) -> pd.DataFrame:
    response_df = pd.concat([response_df_excel,response_df_word], ignore_index=True)
    response_df.reset_index(drop=True, inplace=True)
    response_df.sort_values(by="Similarity", ascending=False, inplace=True)        
    response_df = response_df.head(NUM_RESPONSES_OUTPUT)

    return response_df

def select_highest_similarity_char_limit(response_df_word: pd.DataFrame, response_df_excel: pd.DataFrame) -> pd.DataFrame:

    # Combine and sort the dataframes
    response_df = pd.concat([response_df_excel, response_df_word], ignore_index=True)
    response_df.reset_index(drop=True, inplace=True)
    response_df.sort_values(by="Similarity", ascending=False, inplace=True)

    # Initialize variables to track the total chars
    total_chars = 0
    selected_indices = []

    for index, row in response_df.iterrows():
        response_chars = len(row["Response"])
        if total_chars + response_chars > MAX_OUTPUT_CHAR:
            break  # Stop adding responses if adding this response would exceed the character limit
        total_chars += response_chars
        selected_indices.append(index)

    # print(f"Total characters in df: {total_chars}")
    logging.info(f"Total characters in df: {total_chars}")

    # Create a DataFrame with the selected responses
    selected_df = response_df.iloc[selected_indices]

    return selected_df

def retrieve_documents(query, options, context) -> pd.DataFrame: 
    config = options.get('config', None)
    index_choice = config.get('index_choice', None)
    
    if index_choice == "word_and_pdf":
        response_df = query_index_word_and_pdf(query)
    elif index_choice == 'excel':
        response_df = query_index_excel(query)
    
    elif index_choice == 'both':
        response_df_word_and_pdf = query_index_word_and_pdf(query)
        # print(f'\nWord and pdf:\n')
        # print(response_df_word_and_pdf[['Similarity','file name']].head(NUM_NEIGHBORS_WORD))
        logging.info(f"Query.py - {response_df_word_and_pdf[['Similarity','file name','Response']].head(NUM_NEIGHBORS_WORD)}")
        
        response_df_excel = query_index_excel(query)
        # print(f'\nExcel:')
        # print(response_df_excel[['Similarity','file name']].head(NUM_NEIGHBORS_EXCEL))
        logging.info(f"Query.py - {response_df_excel[['Similarity','file name','Response']].head(NUM_NEIGHBORS_EXCEL)}")
        
        response_df = select_highest_similarity_max_responses(response_df_word_and_pdf,response_df_excel)
   
    elif index_choice == 'both_char_limit':
        response_df_word_and_pdf = query_index_word_and_pdf(query)
        #print(f'\nWord and pdf:')
        #print(response_df_word_and_pdf[['Similarity','file name']].head(NUM_NEIGHBORS_WORD))
        
        response_df_excel = query_index_excel(query)
        #print(f'\nExcel:')
        #print(response_df_excel[['Similarity','file name']].head(NUM_NEIGHBORS_EXCEL))
        
        response_df = select_highest_similarity_char_limit(response_df_word_and_pdf,response_df_excel)
    else: 
        response_df = pd.DataFrame()
        #print("Invalid index choice")
        raise ValueError("Invalid index choice")

    return response_df

def call_api(query, options, context = None) -> dict:

    response_df = retrieve_documents(query, options, context)
    
    if response_df.empty:
        output = "Invalid index choice"
    else: 
        output = '\n'.join(response_df.apply(lambda x: f'["{x["file name"]}", "{x["Response"]}"]\n', axis=1).tolist())


    result = {
        "output": output,
    }
    
    return result

if __name__ == "__main__":
    
    options = {"config": {"index_choice": "both"}}
    
    while True:
        query = input("Enter your query: ")    
            # test if query is empty:
        if query == "":
            #print("Empty query, please enter a query.")
            break
        
        logging.info(f"Query.py - Query: {query}")
        response = call_api(query, options, None)
        # response = call_api(query, {"config": {"index_choice": "excel"}}, None)

        logging.info(f"Query.py - Retrieved documents: {response.get('output')}")


        # print("\nWORD AND PDF:")
        # response_df = query_index_word_and_pdf(query)
        # print(response_df.head())


        # print("\nEXCEL:")
        # response_df = query_index_excel(query)
        # print(response_df.head())
        # print("\n")





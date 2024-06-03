# see: https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings#generative-ai-get-text-embedding-python_vertex_ai_sdk
import json
import time
import pandas as pd
from vertexai.language_models import TextEmbeddingModel

from document_loader import get_file_content, get_files
from splitters import chunk_file
from classes import File, Chunk
from tqdm import tqdm
import math
import pandas as pd

TOKEN_LIMIT = 20000
BATCH_SIZE = 250


def embed_question_local(filename: str,output_filename : str) -> None:
    """Embeds questions from a JSON file using adaptive batch sizes."""
    # read json file
    with open(filename, 'r') as f:
        data = json.load(f)

    # Initialize the model
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

    # get questions
    questions = [d['Question'] for d in data]

    # add id 
    for i, d in enumerate(data):
        d['id'] = i
        
    embeddings = []
    start_index = 0
    while start_index < len(questions):
        # Calculate batch size based on token limit
        token_count = 0
        batch_size = 0
        for question in questions[start_index:]:
            token_count += len(question.split())  # Approximate token count
            if token_count > TOKEN_LIMIT:
                break
            if batch_size == BATCH_SIZE:
                break
            batch_size += 1
        if batch_size == 0:  # Ensure at least one question is processed even if it exceeds the token limit
            batch_size = 1

        # Process batch
        batch_questions = questions[start_index:start_index + batch_size]
        print(f"Processing batch {start_index // batch_size + 1} with {batch_size} questions and token count {token_count}...")
        batch_embeddings = model.get_embeddings(batch_questions, auto_truncate=False)
        embeddings = embeddings + [e.values for e in batch_embeddings]  

        # Move start index for next batch
        start_index += batch_size

        time.sleep(1)  # Add a delay to avoid hitting API restrictions too quickly

    # Add embeddings to data
    for d, emb in zip(data, embeddings):
        d['embedding'] = emb  

    df = pd.DataFrame(data)

    # Save the modified data back to a JSON file
    jsonl_string = df[["id", "Question","Answer","Topic","Sheet name","file name","embedding"]].to_json(orient="records", lines=True)
    with open(output_filename, "w") as f:
        f.write(jsonl_string)

    print(f"Embeddings for {len(data)} questions have been saved to {output_filename}")


def embed_file_local(files: list[File]) -> None:
    """Text embedding with a Large Language Model using batch processing."""
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    for file in tqdm(files, desc="Embedding files"):
        texts = [chunk.text for chunk in file.chunks]
        cum_len = sum(len(text) for text in texts)
        embeddings = []

        try: 
            if cum_len > TOKEN_LIMIT:
                divisions = math.ceil(cum_len / TOKEN_LIMIT)
                # print(f"Procesing {file.file_name} takes {divisions} embedding calls")

                for i in range(divisions):
                    start_idx = i * (len(texts) // divisions)
                    end_idx = (i + 1) * (len(texts) // divisions) if i < divisions - 1 else len(texts)
                    segment = texts[start_idx:end_idx]
                    segment_embeddings = model.get_embeddings(segment)
                    embeddings = embeddings + [e.values for e in segment_embeddings]  
                    # embeddings.extend(segment_embeddings)
            else:
                segment_embeddings = model.get_embeddings(texts,auto_truncate=False)
                embeddings = embeddings + [e.values for e in segment_embeddings]  
            
            for chunk, vector in zip(file.chunks, embeddings):
                chunk.vector = vector
        except Exception as e:
            print(f"Error embedding {file.file_name}: {e}")


def embeddings_to_json(files: list[File], output_filename: str) -> None:
    """Converts the embeddings of a list of files to a JSON file."""
    data = []
    id = 0
    for file in files:
        for chunk in file.chunks:
            data.append({
                "id": id,
                "file_name": file.file_name,
                "chunk_text": chunk.text,
                "embedding": chunk.vector
            })
            id += 1

    
    df = pd.DataFrame(data)

    # Save the modified data back to a JSON file
    jsonl_string = df[["id", "file_name","chunk_text","embedding"]].to_json(orient="records", lines=True)
    with open(output_filename, "w") as f:
        f.write(jsonl_string)

    print(f"Embeddings for {len(files)} files have been saved to {output_filename}")



def get_embeddings_wrapper(texts):
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    
    embs = []
    for i in (range(0, len(texts), BATCH_SIZE)):
        time.sleep(1)  # to avoid the quota error
        result = model.get_embeddings(texts[i : i + BATCH_SIZE])
        embs = embs + [e.values for e in result]
    return embs


if __name__ == "__main__":

    # EXCEL
    output_filename = "1104_QnA_embedded.json"
    embed_question_local('/workspaces/ESGenie/input_data/1104_QnA.json',output_filename)
 
 
    # # WORD and PDF 
    # word_files = get_files(file_type='word',
    #                        parent_folder_list=['2022','2023','2024'],
    #                        num_documents=1000,
    #                        file_overview='2603_organized_files.yaml')
    # print(f"found {len(word_files)} word files")

    # pdf_files = get_files(file_type='pdf',
    #                        parent_folder_list=['2022','2023','2024'],
    #                       num_documents=1000,
    #                       file_overview='2603_organized_files.yaml')
    # print(f"found {len(pdf_files)} pdf files")

    # files = word_files + pdf_files

    # files = get_files(file_type='pdf',parent_folder_list=['other'],num_documents=2)

    # get_file_content(files) 
    # chunk_file(files)
    # embed_file_local(files)
    # embeddings_to_json(files, "pdf_files_with_embeddings.json")


    

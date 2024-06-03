from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate
from query import call_api as retrieve_documents

import logging
import os


# This script is mainly thought to be used as part of the retrieval pipeline for the word 
# and pdf documents. It is used to extract relevant quotes from the documents retrieved, which may 
# reduce the size of the haystack, and increase similiarity.

logging.basicConfig(filename='application.log', level=logging.INFO, 
                    format=f'%(asctime)s - %(levelname)s - %(message)s')


def get_relevant_quotes_template(query, context) -> str:
    # from https://www.promptingguide.ai/techniques/prompt_chaining#prompt-chaining-for-document-qa
    template = """You are a helpful assistant. Your task is to help answer a question given in a document. 
    The first step is to extract quotes relevant to the question from the documents retrieved. 
    The question is delimited by <QUESTION> and <\QUESTION> 
    The documents are delimited by <DOCUMENTS> and <\DOCUMENTS> 
    Please output the list of quotes using <quotes></quotes>. 
    Respond with "No relevant quotes found!" if no relevant quotes were found.

    <DOCUMENTS>
    {context}
    </DOCUMENTS>

    <QUESTION>
    {query}
    </QUESTION>

    """  
    return template



def extract_quotes(query,options,context) -> VertexAI | PromptTemplate:
    logging.info(f"quote_extraction.py - Query: {query}")
    logging.info(f"quote_extraction.py - Options: {options}")

    config = options.get('config', None)
    temperature = config.get('temperature', None)
    prompt_type = config.get('prompt_type',None)

    if temperature is None:
        raise ValueError("Temperature not provided in options")
    
    llm = VertexAI(model_name="gemini-pro", max_output_tokens=1000, temperature=temperature)

    template = get_relevant_quotes_template(query, context)

    # if prompt_type == "generic":
        # template = get_relevant_quotes_template()
    # else:
        # raise ValueError("prompt_type not specified")

    prompt = PromptTemplate.from_template(template)

    chain = prompt | llm

    logging.info(f"quote_extraction.py - Template: {template}")

    return chain.invoke({"query": query, "context": context})


def call_api(query, options, context=None) -> dict:
    # to easily be able to test with promptfoo we have an intermidiate function that exposes an api
    logging.info(f"quote_extraction.py - Query: {query}")

    
    if context == None:
        logging.info("quote_extraction.py - Context is None, retrieving relevant documents.")
        context = retrieve_documents(query, options=options, context=None).get("output", None)
    else:
        logging.info("quote_extraction.py - Context is not None") # when context is passed from promptfoo, it passes all vars.
        context = context.get("vars", None)
        context = context.get("context", None)    
    logging.info(f"quote_extraction.py - Context: {context}")
    output = extract_quotes(query=query, options=options, context=context)
    
    result = {
        "output": output,
    }

    logging.info(f"quote_extraction.py - Output: {output}")

    return result


if __name__ == '__main__':

    options = {"config": {"index_choice": "both"
                            ,"temperature": 0.0
                            ,"prompt_type": "improved2"}}
    logging.info(f"quote_extraction.py - options: {options}")

    print(f"Using index: {options['config']['index_choice']}")
    print(f"Using prompt_template: {options['config']['prompt_type']}")
    while True:
        query = input("Enter your query: ")    
            # test if query is empty:
        if query == "":
            print("Empty query, please enter a query.")
            break
        
        response = call_api(query=query, options=options, context=None)
        
            # write the output to a md file
        output = response.get("output", None)
        
        with open("output_llm.md", "w") as f:
            f.write("Question: "+ query + "\n\n")
            f.write("Answer: " + output + "\n\n")
        print("wrote response to output_llm.md")



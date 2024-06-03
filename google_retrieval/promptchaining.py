from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate
from prompt_engineering_classes import SubQuery
from query import retrieve_documents
from subquery import decompose_query_with_examples as get_subquery


import logging
from tqdm import tqdm


logging.basicConfig(filename='application.log', level=logging.INFO, 
                    format=f'%(asctime)s - %(levelname)s - %(message)s')

def QnA_template2() -> str:
    # Added the sub-question template to reduce the number of hallucinations. from 42 to 46 context-faithfulness.
    template = """You are an expert at answering ESG-related questions .

        Before submitting the answer, make sure to follow the checklist below:
        1. If you don't know the answer, just say that you don't know.
        2. List the names of the used references at the bottom of the output. 
        3. Do not make up information that is not provided directly in the retrieved documents.
        4. Make sure that the output is concise and relevant to the given query.
        5. Stick to the wording of the retrieved documents, and do not add anything that is not present in the documents.
        
        You have retrieved some documents to help you answer the question:

        <DOCUMENTS>
        {context}
        </DOCUMENTS>
        
        <Question>
        {query}
        </Question>

        Answer:
        """  
    return template


def QnA_template1() -> str:
    # Added the sub-question template to reduce the number of hallucinations. from 42 to 46 context-faithfulness.
    template = """You are an expert for question-answering tasks.

        Before submitting the answer, make sure to follow the checklist below:
        1. List the names of the used references at the bottom of the output. 
        2. Make sure that the output is relevant to the given query.
        3. If you don't know the answer, just say that you don't know.
        4. Keep the format similar to prior answers given in the retrieved documents.
        5. Keep the answer concise.
        6. Do not make up information that is not provided directly in the retrieved documents.

        You have retrieved some documents to help you answer the question:

        <DOCUMENTS>
        {context}
        </DOCUMENTS>
        
        <Question>
        {query}
        </Question>

        Answer:
        """  
    return template



def question_from_template(query,options,context) -> VertexAI | PromptTemplate:
    logging.info(f"promptchaining.py - Query: {query}")
    logging.info(f"promptchaining.py - Options: {options}")

    config = options.get('config', None)
    temperature = config.get('temperature', None)
    prompt_type = config.get('prompt_type',None)

    if temperature is None:
        raise ValueError("Temperature not provided in options")
    
    llm = VertexAI(model_name="gemini-pro", max_output_tokens=1000, temperature=temperature)

    if prompt_type == "QnA_template":
        template = QnA_template1()
    elif prompt_type == "QnA_template2":
        template = QnA_template2()
    else:
        raise ValueError("prompt_type not specified")

    prompt = PromptTemplate.from_template(template)

    chain = prompt | llm

    logging.info(f"promptchaining.py - Template: {template}")

    return chain.invoke({"query": query, "context": context})


def get_subquery_context(query, options, context=None) -> str:
    logging.info(f"promptchaining.py - Query: {query}")
    subqueries = get_subquery(query,options)
    logging.info(f"promptchaining.py - Subqueries: {subqueries}")

    all_contexts = []  # Initialize an empty list to collect all sub_query_contexts

    # use tqdm to show progress bar
    for subquery in tqdm(subqueries, desc="Processing subqueries"):
        subquery_question = subquery.sub_query
        logging.info(f"promptchaining.py - Subquery: {subquery_question}")

        response_df = retrieve_documents(subquery_question, options=options, context=None) # convert to list
        context_list = response_df.apply(lambda x: f'["{x["file name"]}", "{x["Response"]}"]\n', axis=1).tolist()
        subquery.sub_query_context = context_list

        all_contexts.extend(subquery.sub_query_context)  

    output = '\n'.join(all_contexts)


    return output

def answer_subqueries(query, options, context=None) -> list[SubQuery]:
    logging.info(f"promptchaining.py - Query: {query}")
    subqueries = get_subquery(query,options)
    logging.info(f"promptchaining.py - Subqueries: {subqueries}")

    # use tqdm to show progress bar
    for subquery in tqdm(subqueries, desc="Processing subqueries"):
        subquery_question = subquery.sub_query
        logging.info(f"promptchaining.py - Subquery: {subquery_question}")

        response_df = retrieve_documents(subquery_question, options=options, context=None) # convert to list
        context_list = response_df.apply(lambda x: f'["{x["file name"]}", "{x["Response"]}"]\n', axis=1).tolist()
        subquery.sub_query_context = context_list
        logging.info(f"promptchaining.py - Context: {context_list}")

        output = question_from_template(query=subquery_question, options=options, context=context_list)
        subquery.sub_query_answer = output
        logging.info(f"promptchaining.py - Subqueries: {subquery}")


    # logging.info(f"promptchaining.py - Output: {output}") # this made it crash? 

    return subqueries


def call_api(query, options, context=None) -> dict:
    config = options.get('config', None)
    enable_metrics = config.get('enable_metrics',None)

    subqueries = answer_subqueries(query, options, context)
    
    output = '\n'.join([subquery.QnA_output(enable_metrics=enable_metrics) for subquery in subqueries])

    result = {
        "output": output,
    }

    return result

if __name__ == '__main__':
    print("Starting ESGenie...")

    options = {"config": {"index_choice": "both"
                            ,"temperature": 0.0
                            ,"prompt_type": "QnA_template2"
                            ,"enable_metrics": True}}
    
    logging.info(f"promptchaining.py - options: {options}")

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
            f.write("Question: "+ query + "\n\n----------------\n\n")
            f.write(output + "\n\n")
        print("wrote response to output_llm.md")





    # model.invoke(question)

    # for chunk in model.stream(question):
    #     print(chunk, end="", flush=True)

    # model.batch([question])

    
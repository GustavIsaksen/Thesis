from langchain_openai import ChatOpenAI
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate
from query import call_api as retrieve_documents

import logging
import os

# Get the name of the current script file
script_name = os.path.basename(__file__)

# Configure logging to include the script name in each message
logging.basicConfig(filename='application.log', level=logging.INFO, 
                    format=f'%(asctime)s - %(levelname)s - %(message)s')

def improved_template4() -> str:
    # 2204 - this template was implemented after testing with the front end. It is a CoT prompt with an example
    template = """You are an expert at answering ESG-related questions .

        Your first task is to meticulously break down a given user question into as many distinct, detailed subqueries as needed to thoroughly answer the original question. Each subquery should explore a unique facet or requirement of the main query. 
        Ensure to cover all bases and leave no stone unturned. Remember, if there are acronyms or terms you do not recognize, maintain their original form. Quesiton: 

        <Question>
        {query}
        </Question>

        After breaking down the question, look at the retrieved context to help you answer the question:

        <context>
        {context}
        </context>

        Based on the provided context, answer each quesiton seperately.

        Before submitting the answer, make sure to follow the checklist below:
        1.	If you don't know the answer, just say that you don't know.
        2.	List the names of the used references at the bottom of the output. 
        3.	Do not make up information that is not provided directly in the retrieved documents.
        4.	Make sure that the output is concise and relevant to the given query.
        5.	Structure your answer by putting together sentences from the retrieved documents, without modifying the sentences.
        6.	Stick to the wording of the sentences in the retrieved documents. 
        7.	Do not add anything that is not present in the documents.
        8.	Do not include quotations
        

        Structure the output so that each question is answered in a separate section. Do not print all the subquestions identified at the start of the output.
        Here is an example of what the output should look like: 

        <Question>
        </Question>
        
        <OUPTUT> 


        References:
        <\OUPTUT>
        

        """  
    return template

def improved_template3() -> str:
    # 
    template = """You are an assistant for question-answering tasks.

        Before submitting the answer, make sure to follow the checklist below:
        1. If you don't know the answer, just say that you don't know.
        2. List references at the bottom of the output. 
        3. Make sure that the output is relevant to the given query.
        4. Do not make up information that is not provided directly in the context.

        You have retrieved some context to help you answer the question:

        <DOCUMENTS>
        {context}
        </DOCUMENTS>
        
        Break the following question into sub-questions, and then answer them seperately. 

        <Question>
        {query}
        </Question>

        """  
    return template

def improved_template2() -> str:
    # Added the sub-question template to reduce the number of hallucinations. from 42 to 46 context-faithfulness.
    template = """You are an assistant for question-answering tasks.

        Before submitting the answer, make sure to follow the checklist below:
        1. List references at the bottom of the output. 
        2. Make sure that the output is relevant to the given query.
        3. If you don't know the answer, just say that you don't know.
        4. Keep the format similar to prior answers given in the context.
        5. Keep the answer concise.
        6. Do not make up information that is not provided directly in the context.

        You have retrieved some context to help you answer the question:

        <DOCUMENTS>
        {context}
        </DOCUMENTS>
        
        Break the following question into sub-questions, and then answer them seperately. 

        <Question>
        {query}
        </Question>

        """  
    return template

def improved_template() -> str:
    template = """You are an assistant for question-answering tasks.

        Before submitting the answer, make sure to follow the checklist below:
        1. List references at the bottom of the output. 
        2. Make sure that the output is relevant to the given query.
        3. If you don't know the answer, just say that you don't know.
        4. Keep the format similar to prior answers given in the context.
        5. Keep the answer concise.
        6. Do not make up information that is not provided directly in the context.

        You have retrieved some context to help you answer the question:

        <DOCUMENTS>
        {context}
        </DOCUMENTS>
        
        Answer the following question: {query}

        """  
    return template

def get_template_from_langchain() -> str:
    template = """    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise. List references at the bottom of the output.
    Question: {query} 
    Context: {context} 
    Answer:"""
    return template

def get_template() -> str:
    # TODO: Need to describe the context in the start template
    # Note: The ESG specialist  was removed from the template as it was causing the LLM to draw on its closed-book knowledge rather than the context provided.
    
    template = """You are an ESG specialist . Answer the following question: {query}

        <DOCUMENTS>
        {context}
        </DOCUMENTS>

        Here is a checklist for the output response:
        1. Make sure to list the name of each document that was used as reference. 
        2. Make sure that the output is relevant to the given query.
        3. If you are unsure about the response, just say that you do not have the neeccessary information to answer the question. """

    return template



def question_from_template(query,options,context) -> VertexAI | PromptTemplate:
    logging.info(f"llm.py - Query: {query}")
    logging.info(f"llm.py - Options: {options}")

    config = options.get('config', None)
    temperature = config.get('temperature', None)
    prompt_type = config.get('prompt_type',None)
    model_type = config.get('model_type',None)

    if temperature is None:
        raise ValueError("Temperature not provided in options")



    if prompt_type == "generic":
        template = get_template_from_langchain()
    elif prompt_type == "esg_specialist":
        template = get_template()
    elif prompt_type == "improved":
        template = improved_template()
    elif prompt_type == "improved2":
        template = improved_template2()
    elif prompt_type == "improved3":
        template = improved_template3()
    elif prompt_type == "improved4":
        template = improved_template4()
    else:
        raise ValueError("prompt_type not specified")

    if model_type == "gpt-4-turbo":    

        from langchain import hub
        from langchain.agents import AgentExecutor, create_openai_functions_agent
        from langchain_community.tools.tavily_search import TavilySearchResults
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(temperature=temperature, model_name="gpt-4-turbo")

        tools = [TavilySearchResults(max_results=1)]

        prompt = hub.pull("hwchase17/openai-functions-agent")

        print(prompt)

        agent = create_openai_functions_agent(llm, tools, prompt)

        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        agent_executor.invoke({"input": "what is LangChain?"})

    elif model_type == "gemini-pro":
        llm = VertexAI(model_name="gemini-pro", max_output_tokens=1000, temperature=temperature)

        prompt = PromptTemplate.from_template(template)

        chain = prompt | llm

        logging.info(f"llm.py - Template: {template}")
        

        return chain.invoke({"query": query, "context": context})
    else:
        raise ValueError("model_type not specified")

    


def call_api(query, options, context=None) -> dict:

    

    if context == None:
        logging.info("llm.py - Context is None")
        context = retrieve_documents(query, options=options, context=None).get("output", None)
    else:
        logging.info("llm.py - Context is not None") # when context is passed from promptfoo, it passes all vars.
        context = context.get("vars", None)
        context = context.get("context", None)    
    logging.info(f"llm.py - Context: {context}")
    output = question_from_template(query=query, options=options, context=context)
    
    result = {
        "output": output,
    }

    logging.info(f"llm.py - Output: {output}")

    return result


if __name__ == '__main__':
    print("Starting ESGenie...")

    options = {"config": {"index_choice": "both"
                            ,"temperature": 0.0
                            ,"prompt_type": "improved2"
                            ,'model_type': 'gpt-4-turbo'}}
    
    logging.info(f"llm.py - options: {options}")

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




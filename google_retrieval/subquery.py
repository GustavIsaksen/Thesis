import logging

from langchain_google_vertexai import VertexAI
from langchain.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import MessagesPlaceholder

from prompt_engineering_classes import SubQuery

import uuid
from typing import Dict, List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

def get_sub_question_template2() -> str:
    system = """You are an expert at converting complex user questions into detailed database queries. \
    Your task is to meticulously break down a given user question into as many distinct, detailed subqueries as needed \
    to thoroughly answer the original question. Each subquery should explore a unique facet or requirement of the main query. \
    Ensure to cover all bases and leave no stone unturned. \
    Remember, if there are acronyms or terms you do not recognize, maintain their original form."""

    return system


def get_sub_question_template() -> str:
    system = """You are an expert at converting user questions into database queries. \

    Perform query decomposition. Given a user question, break it down into distinct sub questions that \
    you need to answer in order to answer the original question.

    If there are acronyms or words you are not familiar with, do not try to rephrase them."""
    
    return system

def create_examples_list() -> list:
    examples = []

    question = "asdf."
    queries = [
        SubQuery(sub_query="asfd", sub_query_context=["",""], sub_query_answer=""),
        SubQuery(sub_query="asdf2?", sub_query_context=["",""], sub_query_answer=""),
        SubQuery(sub_query="asdf23.", sub_query_context=["",""], sub_query_answer=""),
        SubQuery(sub_query="asdf4.", sub_query_context=["",""], sub_query_answer=""),
        SubQuery(sub_query="asdf5.", sub_query_context=["",""], sub_query_answer=""),
        SubQuery(sub_query="asdf.", sub_query_context=["",""], sub_query_answer=""),
    ]
    examples.append({"input": question, "tool_calls": queries})


    question = "asdf."
    queries = [
        SubQuery(sub_query="asdf?", sub_query_context=["",""], sub_query_answer=""),
        SubQuery(sub_query="asdf?2", sub_query_context=["",""], sub_query_answer=""),
        SubQuery(sub_query="asdf?3", sub_query_context=["",""], sub_query_answer=""),
    ]
    examples.append({"input": question, "tool_calls": queries})
    
    return examples


def tool_example_to_messages(example: Dict) -> List[BaseMessage]:
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    openai_tool_calls = []
    for tool_call in example["tool_calls"]:
        openai_tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": tool_call.__class__.__name__,
                    "arguments": tool_call.json(),
                },
            }
        )
    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": openai_tool_calls})
    )
    tool_outputs = example.get("tool_outputs") or [
        "This is an example of a correct usage of this tool. Make sure to continue using the tool this way."
    ] * len(openai_tool_calls)
    for output, tool_call in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages

def decompose_query_with_examples(query,options,context = None) -> list[SubQuery]:

    # TODO: Doesnt work, as the subQuery class cant be initialized with context and answers in the examples. Not sure how the tools would understand this if filled out.


    config = options.get('config', None)
    temperature = config.get('temperature', None)

    system = get_sub_question_template()
    # temperature = options.get("temperature", 0.5)
    
    example_msgs = [msg for ex in create_examples_list() for msg in tool_example_to_messages(ex)]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("examples", optional=True),
            ("human", "{question}"),
        ]
    )

    # llm = VertexAI(model_name="gemini-pro", max_output_tokens=1000, temperature=temperature) 
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=temperature)

    llm_with_tools = llm.bind_tools([SubQuery]) # TODO: bind_tools is not supported for vertexAI
    parser = PydanticToolsParser(tools=[SubQuery])

    query_analyzer_with_examples = (
        prompt.partial(examples=example_msgs) | llm_with_tools | parser
    )

    
    # query_analyzer = prompt | llm_with_tools | parser

    return query_analyzer_with_examples.invoke({"question": query})

def decompose_query(query,options,context = None) -> list[SubQuery]:

    config = options.get('config', None)
    temperature = config.get('temperature', None)

    system = get_sub_question_template()
    # temperature = options.get("temperature", 0.5)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    # llm = VertexAI(model_name="gemini-pro", max_output_tokens=1000, temperature=temperature) 
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=temperature)

    llm_with_tools = llm.bind_tools([SubQuery]) # TODO: bind_tools is not supported for vertexAI
    parser = PydanticToolsParser(tools=[SubQuery])

    query_analyzer = prompt | llm_with_tools | parser

    return query_analyzer.invoke({"question": query})


if __name__ == '__main__':
    options = {"config": {"index_choice": "both"
                            ,"temperature": 0.0
                            ,"prompt_type": "improved2"}}
    
    logging.info(f"subquery.py - options: {options}")

    print(f"Using index: {options['config']['index_choice']}")
    print(f"Using prompt_template: {options['config']['prompt_type']}")
    while True:
        query = input("Enter your query: ")    
            # test if query is empty:
        if query == "":
            print("Empty query, please enter a query.")
            break
        
        output = decompose_query_with_examples(query=query, options=options, context=None)
        
        with open("output_llm.md", "w") as f:
            f.write("Question: "+ query + "\n\n")
            for sub_question in output:
                f.write("Sub_question: " + sub_question.sub_query + "\n\n")
        print("wrote response to output_llm.md")

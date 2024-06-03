from llm import call_api as llm_call_api

# implements the same logic as the original call_api function in google_retrieval/llm.py 
# but allows for visually distinguishing the two implementations in promptfoo

def call_api(query, options, context) -> dict:
    output = llm_call_api(query=query, options=options, context=context)

    return output
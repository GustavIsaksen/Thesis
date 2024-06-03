from query import call_api


# def retrieve_documents(question: str) -> str:
#     # Calculate embeddings, search vector db...
#     context = call_api(question, options, context)
        
#     return context

import logging

def get_var(var_name, prompt, other_vars):

    logging.info(f'load_context.py - Other_vars {other_vars}')

    query = other_vars['query']

    options = {"config": {"index_choice": "both"}}
    logging.info(f'load_context.py - Options: {options}')
        
    output = call_api(query, options = options, context = None)

    # output = {
    #     "output": "This is the retrieved context for the query."
    # }

    return output

    # In case of error:
    # return {
    #     'error': 'Error message'
    # }


if __name__ == '__main__':
    print(get_var(
        var_name = None, 
        prompt = None,
        other_vars = {
            'query': 'asdf',
         }))
from promptchaining import get_subquery_context
import logging

def get_var(var_name, prompt, other_vars):

    logging.info(f'load_context_prompt_chaining.py - Other_vars {other_vars}')

    query = other_vars['query']

    options = {"config": {"index_choice": "both", "temperature": 0.0}}
    logging.info(f'load_context_prompt_chaining.py - Options: {options}')
        
    output = get_subquery_context(query, options = options, context = None)

    output = {
        "output": output
    }

    return output

    # In case of error:
    # return {
    #     'error': 'Error message'
    # }


if __name__ == '__main__':
    query = 'asdf'

    output = get_var(
        var_name = None, 
        prompt = None,
        other_vars = {
            'query': 'asdf',
         })
    

    # write to output_llm.md
    with open("output_llm.md", "w") as f:
        f.write("Question: "+ query + "\n\n")
        f.write("context:: " + output.get("output") + "\n\n")
    print("wrote response to output_llm.md")

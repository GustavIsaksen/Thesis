from subquery import decompose_query

import logging

def get_var(var_name, prompt, other_vars):

    logging.info(f'load_subqueries.py - Other_vars {other_vars}')

    query = other_vars['general_question']

    options = {"config": {"index_choice": "both"}}
    logging.info(f'load_subqueries.py - Options: {options}')
        
    output = decompose_query(query, options = options, context = None)

    output = {
        "output": output
    }

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
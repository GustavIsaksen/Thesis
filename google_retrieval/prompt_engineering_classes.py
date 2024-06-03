import logging
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field

from ragas.evaluation import Result

import pandas as pd
import pyarrow as pa
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_relevancy,
)

class SubQuery(BaseModel):

    sub_query: str = Field(
        description="A very specific query against the database.",
    )

    sub_query_context: list[str] = Field(
        description="The context retrieved to the specific query.",
    )

    sub_query_answer: str = Field(
        description="The answer to the specific query.",
    )

    evaluation: Optional[dict] = None


    def QnA_output(self,enable_metrics = False):
        if enable_metrics:
            self.evaluate()
            outputstring = f"- **Question: {self.sub_query}**\n\nEvaluation: {self.evaluation}\n\nAnswer: {self.sub_query_answer}\n"
        else:
            outputstring = f"- Question: {self.sub_query}\n\nAnswer: {self.sub_query_answer}\n"

        return outputstring
    
    def evaluate(self):
        try:
            logging.info("Evaluating the subquery")
            result_dict = ragas_evaluate(self.sub_query, self.sub_query_context, self.sub_query_answer).to_pandas().to_dict()

            # result_dict = {
            #     "context_relevancy": result_dict.get("context_relevancy")[0],
            #     "faithfulness": result_dict.get("faithfulness")[0],
            #     "answer_relevancy": result_dict.get("answer_relevancy")[0],
            # }
            result_dict = {
                "faithfulness": result_dict.get("faithfulness")[0],
            }
        except:
            logging.info("Couldn't evaluate the subquery")
            logging.info("subquery: ", self.sub_query, "context: ", self.sub_query_context, "answer: ", self.sub_query_answer)
            result_dict = {
                "context_relevancy": None,
                "faithfulness": None,
                "answer_relevancy": None,
            }

        self.evaluation = result_dict

        


def ragas_evaluate(question, context, answer):
    dataset = {
        "question": question,
        "contexts": context,
        "answer": answer,
    }
    df = pd.DataFrame([dataset])
    hg_dataset = Dataset(pa.Table.from_pandas(df))

    result = evaluate(
        hg_dataset,
        metrics=[
            # context_relevancy,
            faithfulness,
            # answer_relevancy,
        ],
    )

    return result

# create a if name == main block to test the class
if __name__ == "__main__":
    test_query = SubQuery(
        sub_query="How to filter a DataFrame in pandas?",
        sub_query_context=["Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool","bananas are yellow","apples are dataframes"],
        sub_query_answer="You can filter a DataFrame in pandas using the syntax `df[df['column_name'] > value]`."
    )

    print(test_query.QnA_output(enable_metrics=True))
    test_query.evaluate()

    print(test_query.evaluation)
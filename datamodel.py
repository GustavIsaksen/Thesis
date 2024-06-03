import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Optional, Union
import json


class TokenUsage(BaseModel):
    total: int
    prompt: int
    completion: int
    cached: Optional[int] = None

class Metrics(BaseModel):
    score: float
    test_pass_count: Optional[int] = None
    test_fail_count: Optional[int] = None
    assert_pass_count: Optional[int] = None
    assert_fail_count: Optional[int] = None
    total_latency_ms: Optional[int] = None
    tokenUsage: TokenUsage  # Note the camelCase to match JSON structure
    named_scores: Optional[Dict[str, float]] = None
    cost: float

class GradingResult(BaseModel):
    pass_field: bool = Field(..., alias='pass')
    score: float
    namedScores: Dict[str, float]
    comment: Optional[str] = None
    value: Optional[str] = None

class Output(BaseModel):
    text : str
    pass_field: bool = Field(..., alias='pass')
    score: float
    named_scores: Optional[Dict[str, float]] = None
    gradingResult: GradingResult

    class Config:
        populate_by_name = True

class Prompt(BaseModel):
    raw: str
    display: str
    id: str
    provider: str
    metrics: Metrics

    class Config:
        populate_by_name = True

class Head(BaseModel):
    prompts: List[Prompt]
    vars: Optional[List[str]] = None

class AssertItem(BaseModel):
    type: str
    value: Optional[Union[str]] = None  # Assuming 'value' can be either a string or a float, adjust as necessary
    threshold: Optional[float] = None  # Adjusted to accept float values
    metric: str

class Test(BaseModel):
    vars: Optional[Dict[str, str]] = None
    assert_: Optional[List[AssertItem]] = Field(None, alias='assert')  # Use the AssertItem model



class Body(BaseModel):
    test: Test
    outputs: List[Output]

class DataModel(BaseModel):
    head: Head
    body: List[Body]
    
    def tabular_df(self):

        # except ValidationError as e:
        #     print(f"Validation error: {e}")
        # Placeholder for flattened data
        flattened_data = []

        for body in self.body:
            query = body.test.vars.get("query"),
            context = body.test.vars.get("context"),

            for output in body.outputs:
                # Flatten the grading_result's named_scores into the output data structure
                flat_output = {
                    "query": query,
                    "context": context,
                    "text" : output.text,
                    "pass_field": output.pass_field,
                    # "score": output.score, # DEV: Seems like this field is broken?
                    "comment": output.gradingResult.comment,
                    **output.gradingResult.namedScores  # Unpack named_scores directly
                }

                # Safely access the first assert_ item if it exists
                if body.test.assert_ and len(body.test.assert_) > 0:
                    first_assert = body.test.assert_[0]
                    flat_output.update({
                        "value": first_assert.value,
                        # Include other fields from AssertItem as needed, e.g., "threshold": first_assert.threshold
                    })

                flattened_data.append(flat_output)

        # Now flattened_data contains all the outputs flattened out, ready for DataFrame conversion
        df = pd.DataFrame(flattened_data)
        return df

        
def test_data_model(sample_data):
    try:
        data_model = DataModel.parse_obj(sample_data)
        print("Data model loaded successfully!")
        print(data_model)
    except ValueError as e:
        print("Value error:", e)

    return data_model

def load_example_data():
    pass


# Run the test function
if __name__ == "__main__":

    

    json_file_path = '/asfd.json'

    with open(json_file_path) as f:
        data = json.load(f)

    data_model = DataModel.parse_obj(data)

    df = data_model.tabular_df()
    
    print(df['text'])
    
    # test_data_model(sample_data)
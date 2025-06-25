# Author: Lau Tsz Yeung Anson
# Contact: s11327605@gm.cyut.edu.tw/tylau70242@gmail.com
# Updated_Date: 2025-06-21
import time
from ollama import Client
import pandas as pd
from pydantic import BaseModel
import json

from dotenv import load_dotenv
import os
load_dotenv()

# JSON Schema
# You can add more key in this part, to fit different task.
class Result(BaseModel):
  response: str

# Get response from LLM
def LLM(client: Client, model: str, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
    ]

    # 1) Debug: print model, lengths
    print(f">>> LLM call starting (model={model})")
    print(f"    system_prompt length: {len(system_prompt)} chars")
    print(f"      user_prompt length: {len(user_prompt)} chars")

    start = time.time()
    try:
        response = client.chat(
            model=model,
            messages=messages,
            options={"seed": 42, "temperature": 0},
            format=Result.model_json_schema(),
        )
    except Exception as e:
        duration = time.time() - start
        print(f"!!! LLM call ERROR after {duration:.1f}s: {e}")
        traceback.print_exc()
        raise
    else:
        duration = time.time() - start
        print(f"<<< LLM call completed in {duration:.1f}s")

    content = response["message"]["content"]
    return content

# Extract JSON from response
def extract_json(response: str):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return json.dumps({"Error": f"'{response}' is not a valid JSON"})

def main(client: Client, model: str):
    data = pd.read_csv("demo.csv")
    num_files = len(data)
    result = []
    start_time = time.time()

    for i in range(num_files):
        # Change system prompt and user prompt here
        system_prompt = "You are a helpful assistant, name Llama."
        user_prompt = data['question'].iloc[i]
        # You can comment out the print statement once you've confirmed the code works as expected.
        print(user_prompt)
        
        # Generate LLM response
        response = LLM(client, model, system_prompt, user_prompt)
        
        # Extract JSON from response
        json_response = extract_json(response)

        # Append result and mark as successful
        result.append(json_response)
        
        # You can comment out the print statement once you've confirmed the code works as expected.
        print(json_response)

    df_result = pd.DataFrame(result)
    df_result.to_json('result.json', orient='records', indent=4)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    # Change host and model here
    # M416_3090, M416_3090ti, M416_4090
    client = Client(host=os.getenv("M416_3090"))
    model = "llama3.1:8b-instruct-fp16"
    main(client, model)

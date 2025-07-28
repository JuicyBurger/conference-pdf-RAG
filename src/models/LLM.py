# Author: Lau Tsz Yeung Anson
# Contact: s11327605@gm.cyut.edu.tw/tylau70242@gmail.com
# Updated_Date: 2025-06-21
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
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
def LLM(
    client: Client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    options: dict = None,
    timeout: float = None,
    raw: bool = False
) -> str:
    opts = {} if options is None else options.copy()
    # By default deterministic:
    if not opts and not raw:
        opts = {"seed": 42, "temperature": 0}
    # If raw=True and no opts, we default to creative
    if raw and not opts:
        opts = {"temperature": 0.4}

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
    ]

    print(f">>> LLM call starting (model={model})")
    print(f"    system_prompt: {len(system_prompt)} chars")
    print(f"      user_prompt: {len(user_prompt)} chars")
    print(f"        options: {opts}")
    if timeout:
        print(f"    timeout: {timeout}s")

    start = time.time()
    try:
        if timeout:
            # run in a thread so we can time out
            with ThreadPoolExecutor(max_workers=1) as exec:
                future = exec.submit(
                    client.chat,
                    model=model,
                    messages=messages,
                    options=opts,
                    format=Result.model_json_schema()
                )
                response = future.result(timeout=timeout)
        else:
            response = client.chat(
                model=model,
                messages=messages,
                options=opts,
                format=Result.model_json_schema()
            )
    except FuturesTimeout:
        raise TimeoutError(f"LLM call timed out after {timeout} seconds")
    except Exception as e:
        dur = time.time() - start
        print(f"!!! LLM call ERROR after {dur:.1f}s: {e}")
        traceback.print_exc()
        raise
    else:
        dur = time.time() - start
        print(f"<<< LLM call completed in {dur:.1f}s")

    # Extract the assistant's content
    print(response["message"]["content"])
    return extract_json(response["message"]["content"])

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

import json
import torch
from tqdm import tqdm
import pandas as pd
from transformers import pipeline

jokes = pd.read_csv(open("./data/jokes.csv"))

token = "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"
model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    token=token,
)

rationales = []

for joke in tqdm(jokes["joke_text"]):
    messages = [
        {
            "role": "system",
            "content": "\n".join(
                [
                    "You are an evaluator for jokes.",
                    "You need to provide a rationale for why the joke is funny.",
                    "Your explanation should be less than 3 sentences.",
                ]
            ),
        },
        {
            "role": "user",
            "content": "Explain why this joke is funny. ```" + joke + "```",
        },
    ]
    outputs = pipe(
        messages,
        max_new_tokens=1024,
    )

    rationale = outputs[0]["generated_text"][-1]["content"]
    rationales.append(rationale)

json.dump({"rationales": rationales}, open("./data/rationales.json", "w"))

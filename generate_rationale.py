import json
from tqdm import tqdm
import pandas as pd
import requests

url = "http://localhost:11434/api/chat"


def llama3(model, messages):
    data = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=data)
    print(response.json())

    return response.json()["message"]["content"]


if __name__ == "__main__":
    jokes = pd.read_csv(open("./data/jokes.csv"))
    rationales = []
    model = "llama3.1:8b"

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
        rationale = llama3(model, messages)
        rationales.append(rationale)

    json.dump({"rationales": rationales}, open("./data/rationales.json", "w"))

import requests
import json

url = 'http://8.130.161.29/v1/embeddings'
with open("/Users/jiannan.yinthoughtworks.com/OTR/otr-performance/sweep/sweepai/utils/test_snippet.txt", "r") as f:
    text = f.read()
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}
data = {
    "max_tokens": 8000,
    "messages": [
        {
            # "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n" + text +"\n\n### Response:",
            "content": text,
            "role": "user"
        }
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())




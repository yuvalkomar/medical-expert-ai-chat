from dotenv import load_dotenv

load_dotenv()

import boto3

client = boto3.client(
    "bedrock-runtime",
    region_name="eu-north-1",
)

response = client.converse(
    modelId="eu.amazon.nova-pro-v1:0",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "text": "hi"
                }
            ],
        }
    ],
)

print(response["output"]["message"]["content"][0]["text"])

import boto3
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    client = boto3.client(
        "bedrock-runtime",
        region_name="eu-north-1",
    )

    response = client.converse(
        modelId="eu.amazon.nova-pro-v1:0",
        messages=[
            {
                "role": "user",
                "content": [{"text": "hi"}],
            }
        ],
    )

    print(response["output"]["message"]["content"][0]["text"])


if __name__ == "__main__":
    main()

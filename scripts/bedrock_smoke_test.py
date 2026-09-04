"""Run one explicit, real AWS Bedrock streaming request using the project's settings."""

import argparse
import asyncio

from backend.app.config import get_settings
from backend.app.llm.base import ChatTurn
from backend.app.llm.factory import create_provider
from backend.app.prompts import MEDICAL_SYSTEM_PROMPT


async def run(question: str) -> None:
    settings = get_settings()
    if settings.llm_provider != "bedrock":
        raise SystemExit("Set LLM_PROVIDER=bedrock in .env before running this script.")

    provider = create_provider(settings)
    messages = [ChatTurn(role="user", content=question)]
    async for chunk in provider.stream(MEDICAL_SYSTEM_PROMPT, messages):
        print(chunk, end="", flush=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one streaming request to AWS Bedrock.")
    parser.add_argument(
        "question",
        nargs="?",
        default="In one sentence, what is hydration?",
    )
    args = parser.parse_args()
    asyncio.run(run(args.question))


if __name__ == "__main__":
    main()

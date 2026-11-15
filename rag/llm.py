"""Single factory for the Cohere Command LLM used by all workflows."""
from __future__ import annotations

from llama_index.core.types import PydanticProgramMode

from rag.embedding import _require_env


def get_llm(cfg: dict, temperature: float | None = None):
    """pydantic_program_mode=LLM: structured_predict puts the JSON schema in the prompt and parses the
    answer with Pydantic. Function calling is NOT used because llama-index-llms-cohere 0.10.0 flattens
    nested objects in tool schemas (lists of objects come back as lists of strings)."""
    from llama_index.llms.cohere import Cohere

    return Cohere(api_key=_require_env("COHERE_API_KEY"), model=cfg["llm"]["model_name"],
                  temperature=cfg["llm"].get("temperature", 0.1) if temperature is None else temperature,
                  pydantic_program_mode=PydanticProgramMode.LLM)


async def retry_on_rate_limit(make_call, attempts: int = 4, base_delay: float = 6.0):
    """Await make_call(); on Cohere 429 (TooManyRequestsError) wait 6s/12s/24s and retry.
    Any other error, or the last 429, is raised to the caller."""
    import asyncio
    import logging

    for i in range(attempts):
        try:
            return await make_call()
        except Exception as e:
            if type(e).__name__ != "TooManyRequestsError" or i == attempts - 1:
                raise
            delay = base_delay * 2 ** i
            logging.getLogger("rag.llm").warning("Cohere rate limit (429), retry %d/%d in %.0fs", i + 1, attempts - 1, delay)
            await asyncio.sleep(delay)

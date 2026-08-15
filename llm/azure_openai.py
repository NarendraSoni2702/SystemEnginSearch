import random
import time

import numpy as np

from openai import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    OpenAI,
    RateLimitError,
)


class AzureOpenAIClient:

    def __init__(
            self,
            endpoint: str,
            api_key: str,
            max_retries: int = 4,
            retry_base_delay: float = 2.0,
            timeout: float = 300.0,
    ):

        endpoint = endpoint.rstrip("/")

        self.client = OpenAI(
            api_key=api_key,
            base_url=(
                f"{endpoint}/openai/v1/"
            ),
            timeout=timeout,
            max_retries=0,
        )

        self.max_retries = (
            max_retries
        )

        self.retry_base_delay = (
            retry_base_delay
        )

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    def embeddings(
            self,
            texts: list[str],
            deployment: str,
    ) -> np.ndarray:

        if not texts:

            return np.empty(
                (0, 0),
                dtype=np.float32,
            )

        response = self._with_retry(
            lambda:
            self.client.embeddings.create(
                model=deployment,
                input=texts,
            )
        )

        items = sorted(
            response.data,
            key=lambda item: item.index,
        )

        vectors = [
            item.embedding
            for item in items
        ]

        if not vectors:

            raise RuntimeError(
                "Azure OpenAI returned "
                "no embeddings."
            )

        return np.asarray(
            vectors,
            dtype=np.float32,
        )

    # ========================================================
    # RESPONSES API
    # ========================================================

    def response(
            self,
            instructions: str,
            input_text: str,
            deployment: str,
            temperature: float,
            max_output_tokens: int,
    ) -> str:

        response = self._with_retry(
            lambda:
            self.client.responses.create(
                model=deployment,

                instructions=(
                    instructions
                ),

                input=input_text,

                temperature=temperature,

                max_output_tokens=(
                    max_output_tokens
                ),

                store=False,
            )
        )

        answer = getattr(
            response,
            "output_text",
            None,
        )

        if answer:
            return answer.strip()

        # Defensive fallback in case the SDK
        # response does not expose output_text.
        return self._extract_output_text(
            response
        )

    # ========================================================
    # RETRY
    # ========================================================

    def _with_retry(
            self,
            operation,
    ):

        last_error = None

        for attempt in range(
                self.max_retries + 1
        ):

            try:

                return operation()

            except (
                    RateLimitError,
                    APIConnectionError,
                    APITimeoutError,
            ) as error:

                last_error = error

                if attempt >= (
                        self.max_retries
                ):
                    raise

                delay = (
                        self.retry_base_delay
                        * (2 ** attempt)
                )

                # Small random jitter prevents
                # synchronized retries.
                delay += random.uniform(
                    0,
                    0.5,
                )

                print(
                    f"\nAzure OpenAI request "
                    f"failed. Retrying in "
                    f"{delay:.1f}s..."
                )

                time.sleep(delay)

            except APIStatusError as error:

                last_error = error

                status = (
                    error.status_code
                )

                # Retry only errors that are
                # normally transient.
                retryable = (
                        status == 408
                        or status == 409
                        or status == 429
                        or status >= 500
                )

                if (
                        not retryable
                        or attempt >= (
                        self.max_retries
                )
                ):

                    raise

                delay = (
                        self.retry_base_delay
                        * (2 ** attempt)
                )

                delay += random.uniform(
                    0,
                    0.5,
                )

                print(
                    f"\nAzure OpenAI returned "
                    f"HTTP {status}. "
                    f"Retrying in "
                    f"{delay:.1f}s..."
                )

                time.sleep(delay)

        raise RuntimeError(
            f"Azure OpenAI request failed: "
            f"{last_error}"
        )

    # ========================================================
    # FALLBACK OUTPUT PARSER
    # ========================================================

    @staticmethod
    def _extract_output_text(
            response,
    ) -> str:

        parts = []

        output = getattr(
            response,
            "output",
            [],
        )

        for item in output:

            content_items = getattr(
                item,
                "content",
                [],
            )

            for content in (
                    content_items
            ):

                text = getattr(
                    content,
                    "text",
                    None,
                )

                if text:
                    parts.append(text)

        answer = "\n".join(
            parts
        ).strip()

        if not answer:

            raise RuntimeError(
                "Azure OpenAI returned "
                "an empty response."
            )

        return answer

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self.client.close()
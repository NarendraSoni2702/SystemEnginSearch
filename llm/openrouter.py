import httpx
import numpy as np


class OpenRouterClient:

    def __init__(
            self,
            api_key: str,
            base_url: str,
            connect_timeout: float = 20.0,
            read_timeout: float = 300.0,
            write_timeout: float = 300.0,
            pool_timeout: float = 20.0,
    ):
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set."
            )

        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
            pool=pool_timeout,
        )

        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Local RAG File Assistant",
            },
        )

        self.base_url = base_url.rstrip("/")

    def close(self):
        self.client.close()

    def embeddings(
            self,
            texts: list[str],
            model: str,
    ) -> np.ndarray:

        if not texts:
            return np.empty(
                (0, 0),
                dtype=np.float32,
            )

        response = self.client.post(
            f"{self.base_url}/embeddings",
            json={
                "model": model,
                "input": texts,
            },
        )

        self._raise_for_api_error(
            response,
            "Embedding",
        )

        data = response.json()

        items = sorted(
            data.get("data", []),
            key=lambda item: item.get("index", 0),
        )

        if not items:
            raise RuntimeError(
                "OpenRouter did not return embeddings."
            )

        vectors = []

        for item in items:
            embedding = item.get("embedding")

            if embedding is None:
                raise RuntimeError(
                    "Missing embedding in API response."
                )

            vectors.append(embedding)

        return np.asarray(
            vectors,
            dtype=np.float32,
        )

    def chat(
            self,
            system_prompt: str,
            user_prompt: str,
            model: str,
            temperature: float,
            max_tokens: int,
    ) -> str:

        response = self.client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

        self._raise_for_api_error(
            response,
            "LLM",
        )

        data = response.json()

        choices = data.get("choices", [])

        if not choices:
            raise RuntimeError(
                "OpenRouter did not return an answer."
            )

        answer = (
            choices[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not answer:
            raise RuntimeError(
                "OpenRouter returned an empty answer."
            )

        return answer

    @staticmethod
    def _raise_for_api_error(
            response: httpx.Response,
            operation: str,
    ):
        if response.status_code == 200:
            return

        try:
            error = response.json()
        except Exception:
            error = response.text

        raise RuntimeError(
            f"{operation} API error "
            f"{response.status_code}: {error}"
        )
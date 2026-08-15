from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5",
    input="Explain artificial intelligence in one sentence."
)

print(response.output_text)
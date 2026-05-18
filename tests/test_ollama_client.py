from llm.ollama_client import OllamaClient

client = OllamaClient()

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": "Explain acne in one short paragraph.",
    },
]

response = client.generate(messages)

print(response)
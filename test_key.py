import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

r = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "回复两个字：通了"}],
)
print(r.choices[0].message.content)

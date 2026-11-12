import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

SYSTEM = "你是一名严谨的矿业经济分析师，回答必须给出数字或依据，不确定时明确说不知道。"


while True:
    q = input("\n你问（输入 exit 退出）：")
    if q.strip().lower() in ("exit", "quit", "退出"):
        break

    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": q},
        ],
    )
    print("\n回答：" + r.choices[0].message.content)

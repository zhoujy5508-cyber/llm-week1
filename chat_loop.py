import os
from dotenv import load_dotenv          # ★1
from openai import OpenAI

load_dotenv()                           # ★1 必须在 OpenAI() 之前

client = OpenAI(                        # ★2
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

messages = [{"role": "system", "content": "你是懂矿产能源行业的助手，回答不超过5句话。"}]   # ★3 循环外！只建一次

print("开始对话（输入 exit 退出）")
while True:
    q = input("你：")
    if q.lower() == "exit":             # ★8
        break

    messages.append({"role": "user", "content": q})          # ★4 你的话入列表
    resp = client.chat.completions.create(                   # ★5 整个列表发出去
        model="deepseek-chat",
        messages=messages,
        stream=False,
    )
    answer = resp.choices[0].message.content                 # ★6 从 JSON 挖文字
    print("助手：", answer)

    messages.append({"role": "assistant", "content": answer})  # ★7 它的回答也存回去（灵魂）
    print(f"--- messages {len(messages)} 条，本次送入 token {resp.usage.prompt_tokens}")

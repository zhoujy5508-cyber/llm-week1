import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

MATERIAL = """2026年9月，某省发展改革委印发《关于完善工商业峰谷分时电价机制的通知》，
自2026年11月1日起，峰段电价上浮比例由50%提高至70%，谷段下浮比例由50%调整为60%，
执行范围扩大至电压等级10千伏及以上的全部工商业用户。"""

VAGUE = "帮我分析一下这个政策"

CLEAR = """你是一名矿业政策分析师，服务对象是省内一家锂盐加工企业。

【任务】判断这份政策对该企业用电成本的影响方向与幅度，并给出应对建议。

【材料】
{材料}

【输出格式】
1. 影响方向：利好 / 利空 / 中性（三选一）
2. 影响幅度：高 / 中 / 低，并用一句话说明依据
3. 关键条款：引用原文不超过 30 字
4. 建议：2 条，每条不超过 40 字

只输出上述四点，不要写开场白和总结。"""

def ask(prompt):
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content

for name, p in [("模糊版", VAGUE + "\n\n" + MATERIAL),
                ("清晰版", CLEAR.replace("{材料}", MATERIAL))]:
    print("\n" + "=" * 18, name, "=" * 18)
    print(ask(p))

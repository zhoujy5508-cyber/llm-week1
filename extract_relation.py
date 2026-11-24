import os
from dotenv import load_dotenv
from openai import OpenAI
import json
# 1. 基础客户端配置（从系统环境变量读取，保证安全性）
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
# 2. 实证测试文本：关键矿产出口管制政策片段
test_text = "商务部、海关总署决定对镓、锗相关物项实施出口管制，未经许可不得出口，以维护国家安全和战略利益。"

# 3. 对照组 A：Zero-shot 提示词模板
prompt_zero_shot = f"""你是一名关键矿产政策研究专家。请从给定的政策文本中抽取关系三元组。
输出必须是严格的 JSON 格式，不要包含任何闲聊、解释或 markdown 标记。

待分析文本：{test_text}
输出格式遵循：{{"triplets": [{{"head": "实体1", "relation": "关系", "tail": "实体2"}}]}}
"""

# 4. 对照组 B：Few-shot 提示词模板（注意：f-string 内部 JSON 示例必须用 {{ 和 }} 转义）
prompt_few_shot = f"""你是一名关键矿产政策研究专家。请从给定的政策文本中抽取关系三元组。
输出必须是严格的 JSON 格式，不要包含任何闲聊、解释或 markdown 标记。

示例：
输入：美国能源部宣布对锂、钴等电池关键原料实施补贴限制，对全球动力电池供应链造成直接冲击。
输出：{{"triplets": [{{"head": "美国能源部", "relation": "实施限制", "tail": "锂钴关键原料"}}, {{"head": "限制政策", "relation": "冲击", "tail": "动力电池供应链"}}]}}

待分析文本：{test_text}
输出格式遵循：{{"triplets": [{{"head": "实体1", "relation": "关系", "tail": "实体2"}}]}}
"""


def extract(prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",  # 填入你当前使用的模型
        messages=[
            {"role": "system", "content": "你是一个严格输出 JSON 的学术数据抽取助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0  # 学术抽取设为 0，确保输出确定性与可复现
    )
    raw = response.choices[0].message.content.strip()

    # 防御性清洗：防止模型意外包裹 markdown 语法标记
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]

    parsed_json = json.loads(raw.strip())
    usage_tokens = response.usage.total_tokens
    return parsed_json, usage_tokens


# 6. 主执行入口：运行 Zero-shot 与 Few-shot 实证对照
if __name__ == "__main__":
    print("====== 1. 运行 Zero-shot 抽取 ======")
    res_zero, token_zero = extract(prompt_zero_shot)
    print("【抽取结果】:")
    print(json.dumps(res_zero, ensure_ascii=False, indent=2))
    print(f"【消耗总 Token】: {token_zero}\n")

    print("====== 2. 运行 Few-shot 抽取 ======")
    res_few, token_few = extract(prompt_few_shot)
    print("【抽取结果】:")
    print(json.dumps(res_few, ensure_ascii=False, indent=2))
    print(f"【消耗总 Token】: {token_few}")
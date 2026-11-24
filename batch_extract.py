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



# 2. 构建微型实证数据集：精选 4 条典型矿产供应链与双碳政策片段
corpus = [
    {
        "id": "doc_01",
        "text": "商务部、海关总署决定对镓、锗相关物项实施出口管制，未经许可不得出口，以维护国家安全和战略利益。"
    },
    {
        "id": "doc_02",
        "text": "生态环境部发布全国碳排放权交易市场新纳入行业方案，将电解铝与水泥行业列入强制减排管控范围。"
    },
    {
        "id": "doc_03",
        "text": "欧洲议会正式通过《关键原材料法案》，旨在降低对第三国锂、稀土等战略矿产的进口依赖度。"
    },
    {
        "id": "doc_04",
        "text": "【故意设计的异常噪声样本】这是一条没有明确政策主体和关系的无效口号文本，用于检验批处理稳定性。"
    }
]

# 3. 抽取 Prompt 模板（采用周一验证收敛效果最好的 Few-shot 模式）
prompt_template = """你是一名关键矿产与双碳政策研究专家。请从给定的政策文本中抽取关系三元组。
输出必须是严格的 JSON 格式，不要包含任何闲聊、解释或 markdown 标记。

示例：
输入：美国宣布对进口大型动力电池关键矿物实施原产地限制政策，削弱了跨国供应链弹性。
输出：{{"triplets": [{{"head": "美国", "relation": "实施限制政策", "tail": "关键矿物"}}, {{"head": "限制政策", "relation": "削弱", "tail": "供应链弹性"}}]}}

待分析文本：{input_text}
输出格式遵循：{{"triplets": [{{"head": "实体1", "relation": "关系", "tail": "实体2"}}]}}
"""


def extract_one(text):
    prompt = prompt_template.format(input_text=text)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个严格输出 JSON 的学术数据抽取助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    raw = response.choices[0].message.content.strip()

    # 清洗 Markdown 反引号
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]

    return json.loads(raw.strip()), response.usage.total_tokens


# 4. 主执行入口：多样本批处理循环与异常日志捕获
if __name__ == "__main__":
    success_results = []
    error_log_path = "error_log.txt"
    total_tokens_spent = 0

    print(f"====== 开始执行批处理：共 {len(corpus)} 条政策语料 ======\n")

    for item in corpus:
        doc_id = item["id"]
        doc_text = item["text"]
        print(f"正在处理样本 [{doc_id}]...")

        try:
            # 尝试抽取三元组
            parsed_data, tokens = extract_one(doc_text)
            total_tokens_spent += tokens

            # 记录成功提取的样本
            success_results.append({
                "doc_id": doc_id,
                "triplets": parsed_data.get("triplets", [])
            })
            print(
                f"  -> [{doc_id}] 抽取成功，提取到 {len(parsed_data.get('triplets', []))} 条三元组 (耗费 {tokens} Tokens)")

        except json.JSONDecodeError as e:
            # 捕获格式解析异常，不让程序崩溃，记录到错误台账
            print(f"  [!] [{doc_id}] 解析失败：JSON 格式异常，已写入错误日志。")
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"样本ID: {doc_id}\n原始文本: {doc_text}\n错误详情: {str(e)}\n{'-' * 40}\n")

        except Exception as e:
            # 捕获网络超时或 API 其他偶发异常
            print(f"  [!] [{doc_id}] 抽取失败：{str(e)}，已写入错误日志。")
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"样本ID: {doc_id}\n原始文本: {doc_text}\n异常原因: {str(e)}\n{'-' * 40}\n")

    print("\n====== 批处理完成 ======")
    print(f"成功处理样本数: {len(success_results)} / {len(corpus)}")
    print(f"累计 Token 消耗: {total_tokens_spent}")

    # 打印第一条成功样本的数据结构（检验规范性）
    if success_results:
        print("\n【样本抽取样例 doc_01】:")
        print(json.dumps(success_results[0], ensure_ascii=False, indent=2))
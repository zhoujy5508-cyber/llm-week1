import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-chat"
CATS = ["锂矿", "煤炭", "油气", "电力", "新能源"]

SYSTEM = (
    "你是财经新闻分类器。只能从以下五类中选一个输出：锂矿、煤炭、油气、电力、新能源。\n"
    "判定规则：按新闻涉及的行业归类，不按数据性质归类。\n"
    "例如：涉及用电量、电网、发电装机的，一律归电力，即使它引用的是宏观统计数据。\n"
    "只输出类别名，不要解释。"
)


# few-shot 示例：注意和测试集措辞不同，避免"背题"
EXAMPLES = [
    {"role": "user", "content": "赣锋锂业宣布扩产，盐湖提锂成本下降。"},
    {"role": "assistant", "content": "锂矿"},
    {"role": "user", "content": "内蒙古大型露天煤矿获批增产。"},
    {"role": "assistant", "content": "煤炭"},
    {"role": "user", "content": "国际油价震荡，炼厂利润承压。"},
    {"role": "assistant", "content": "油气"},
    {"role": "user", "content": "南方电网最大负荷创历史新高。"},
    {"role": "assistant", "content": "电力"},
    {"role": "user", "content": "光伏组件价格战持续，分布式装机放量。"},
    {"role": "assistant", "content": "新能源"},
]

DATA = [
    ("智利锂矿出口关税上调，碳酸锂价格应声上涨。", "锂矿"),
    ("国家发改委批复三个大型煤矿项目，总产能 2000 万吨。", "煤炭"),
    ("中石油在塔里木盆地获高产油气流，日产原油超千吨。", "油气"),
    ("全国全社会用电量同比增长 6.2%，第三产业增速领先。", "电力"),
    ("宁德时代发布新一代储能电池，循环寿命提升 30%。", "新能源"),
    ("江西宜春锂云母矿整治后复产，供应逐步恢复。", "锂矿"),
    ("国际油价跌破 70 美元，地炼开工率降至六成。", "油气"),
    ("某煤电一体化项目获核准，配套 2×100 万千瓦机组。", "电力"),
]

def normalize(text):
    """模型偶尔会多说一句，这里兜底：只认类别名"""
    for c in CATS:
        if c in text:
            return c
    return "未识别"

def classify(news, few_shot=False):
    msgs = [{"role": "system", "content": SYSTEM}]
    if few_shot:
        msgs += EXAMPLES
    msgs.append({"role": "user", "content": news})

    r = client.chat.completions.create(
        model=MODEL,
        messages=msgs,
        temperature=0,          # 分类任务要稳定，别让它自由发挥
    )
    raw = r.choices[0].message.content
    print("  [原始输出]", repr(raw))
    return normalize(r.choices[0].message.content)

def run(few_shot):
    rows, correct = [], 0
    for news, gold in DATA:
        pred = classify(news, few_shot)
        ok = pred == gold
        correct += ok
        rows.append((news, gold, pred, "对" if ok else "错"))
    return correct, rows

for mode, flag in [("zero-shot", False), ("few-shot", True)]:
    correct, rows = run(flag)
    print(f"\n===== {mode}：{correct}/{len(DATA)} 正确 =====")
    for news, gold, pred, mark in rows:
        print(f"[{mark}] {news[:22]}...  标准={gold}  预测={pred}")

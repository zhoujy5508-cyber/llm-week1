import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 自动定位路径并读取根目录 .env 里的 DEEPSEEK_API_KEY
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR.parent / ".env")

STORAGE_DIR = BASE_DIR / "storage"                      # 旋钮 1：读哪份索引（周六对比 256/512/1024 时改这里）
MODEL_DIR = BASE_DIR / "models" / "bge-small-zh-v1.5"   # 本地离线向量模型（必须和建索引时用同一个！）
TOP_K = 3                                               # 旋钮 2：每次捞回最相关的几段原文（可改 3 / 5 对比）

# 2. 从硬盘加载昨天建好的向量索引（几秒秒开，不重新切片）
Settings.embed_model = HuggingFaceEmbedding(model_name=str(MODEL_DIR))
storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
index = load_index_from_storage(storage_context)
retriever = index.as_retriever(similarity_top_k=TOP_K)

# 3. 复用你第 1–2 周最熟的 DeepSeek 接口（透明可控，方便统计 token 与改 Prompt）
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

# 旋钮 3：防幻觉指令（必须只根据检索到的原文答，并标出文件名出处）
SYSTEM_PROMPT = (
    "你是矿产与能源政策研究助手。你只能严格依据下面提供的【参考原文】回答问题：\n"
    "1. 先给出一句话核心结论；\n"
    "2. 分点说明依据，并在每点末尾用【出处：文件名】标明来自哪份文件；\n"
    "3. 如果参考原文里没有提到，必须直接回答「现有语料中未提及」，严禁自行编造数字或条款。"
)

def ask_rag(question: str):
    # ====== 第一阶段：检索（只捞原文，先不问大模型） ======
    nodes = retriever.retrieve(question)
    print(f"\n================ 问题：{question} ================")
    print(f"【第一步 · 召回的 Top-{len(nodes)} 原文片段（先人工看答案在不在里面）】")
    context_parts = []
    for i, node in enumerate(nodes, 1):
        fname = node.metadata.get("file_name", "未知文件")
        score = node.score or 0.0
        snippet = node.text.strip()
        print(f"\n--- 片段 {i} | 相似度={score:.3f} | 出处：{fname} ---")
        print(snippet[:220] + ("..." if len(snippet) > 220 else ""))
        context_parts.append(f"[片段{i}｜出处：{fname}]\n{snippet}")

    # ====== 第二阶段：生成（把捞回的原文拼进 messages 发给 DeepSeek） ======
    context = "\n\n".join(context_parts)
    user_msg = f"【参考原文】\n{context}\n\n【问题】\n{question}"

    resp = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0,                                  # 旋钮 4：研究型问答固定 0，保证可复现
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    answer = resp.choices[0].message.content
    usage = resp.usage
    print("\n【第二步 · DeepSeek 基于原文的回答】")
    print(answer)
    print(f"\n[统计] 输入 prompt_tokens={usage.prompt_tokens}, 输出 completion_tokens={usage.completion_tokens}")

if __name__ == "__main__":
    # 旋钮 5：3 道专门用来“把 RAG 测出毛病”的政策细节题（覆盖：换词语义 / 跨文件 / 长文首尾分离）
    ask_rag("国家现在还强制要求新建光伏或新能源项目必须配储能吗？请给依据。")
    ask_rag("请同时结合分布式光伏管理办法和绿证规则，说明分布式光伏项目如何上网以及能否核发绿证？")
    ask_rag("新修订的矿产资源法对“战略性矿产”有哪些支持政策，是从哪一天开始施行的？")

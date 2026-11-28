"""
W3 周六核心实验：单变量切片对照（chunk_size = 256 / 512 / 1024）
固定向量模型、固定 TOP_K=3、固定同一套 10 条黄金评测集 eval_qa.json，
只改 chunk_size，量化对比：chunk数 / 均长 / 检索命中率 / 上下文长度（token成本代理）。
"""
import csv
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ================= 路径与旋钮区（以后做实验就在这里改） =================
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR.parent / ".env")

DOCS_DIR = BASE_DIR / "docs"
MODEL_DIR = BASE_DIR / "models" / "bge-small-zh-v1.5"
QA_FILE = BASE_DIR / "eval_qa.json"
OUT_CSV = BASE_DIR / "chunk_results.csv"

CHUNK_SIZES = [256, 512, 1024]   # 旋钮 1：要对比的切片大小档位（唯一自变量！）
CHUNK_OVERLAP = 50               # 旋钮 2：相邻块重叠 token（固定不变，保证单变量）
TOP_K = 3                        # 旋钮 3：每次召回几块（固定 3，后面可改 5 做敏感性）
RUN_LLM = False                  # 旋钮 4：False=只测检索层(0元/15秒)；True=顺便调DeepSeek算生成正确率

# 加载本地离线中文向量模型 + 5 份政策原文 + 10 条黄金考题
Settings.embed_model = HuggingFaceEmbedding(model_name=str(MODEL_DIR))
docs = SimpleDirectoryReader(str(DOCS_DIR)).load_data()
qa_list = json.load(open(QA_FILE, encoding="utf-8"))

if RUN_LLM:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")


def evaluate_one_size(chunk_size: int):
    # 1. 按当前 chunk_size 切块并建内存向量索引
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(docs)
    avg_chars = round(sum(len(n.text) for n in nodes) / len(nodes), 1)
    index = VectorStoreIndex(nodes)
    retriever = index.as_retriever(similarity_top_k=TOP_K)

    # 2. 用同一批 10 道黄金题逐题测试检索命中情况
    file_hits, exact_hits, total_ctx_chars = 0, 0, 0
    per_q = {}

    for item in qa_list:
        retrieved = retriever.retrieve(item["question"])
        fnames = [n.metadata.get("file_name", "") for n in retrieved]
        joined_text = "\n".join(n.text for n in retrieved)
        total_ctx_chars += len(joined_text)

        # 判定标准 1：Top-3 里有没有捞到正确出处文件
        f_ok = item["expected_file"] in fnames
        # 判定标准 2（严格）：既捞对文件，又完整包含所有必须命中的原文依据词
        kw_ok = all(kw in joined_text for kw in item["must_contain"])
        hit = f_ok and kw_ok

        file_hits += int(f_ok)
        exact_hits += int(hit)
        per_q[item["id"]] = "✅" if hit else ("🟡仅中文件" if f_ok else "❌漏召回")

    return {
        "chunk_size": chunk_size,
        "num_chunks": len(nodes),
        "avg_chars": avg_chars,
        "file_hit_rate": f"{file_hits}/{len(qa_list)} ({file_hits*10}%)",
        "exact_hit_rate": f"{exact_hits}/{len(qa_list)} ({exact_hits*10}%)",
        "avg_ctx_chars": round(total_ctx_chars / len(qa_list), 0),
        "per_q": per_q,
    }


if __name__ == "__main__":
    print(f"开始单变量切片对照实验（固定 TOP_K={TOP_K}, overlap={CHUNK_OVERLAP}, 题数={len(qa_list)}）...\n")
    results = [evaluate_one_size(cs) for cs in CHUNK_SIZES]

    # 打印汇总对照表
    print(f"{'chunk_size':<10} | {'总块数':<6} | {'块均字数':<8} | {'文件命中率':<12} | {'严格依据命中率':<14} | {'单题Top3均字数(成本)'}")
    print("-" * 88)
    for r in results:
        print(f"{r['chunk_size']:<10} | {r['num_chunks']:<8} | {r['avg_chars']:<10.1f} | {r['file_hit_rate']:<14} | {r['exact_hit_rate']:<16} | {int(r['avg_ctx_chars'])} 字")

    # 打印分题明细表（一眼看出哪类题最挑切片！）
    print("\n【分题命中明细（✅=文件+依据词全中；🟡=中文件但漏关键句；❌=未捞到该文件）】")
    header = f"{'题号':<4} | {'题型':<10} | " + " | ".join(f"{cs:<10}" for cs in CHUNK_SIZES)
    print(header)
    print("-" * len(header))
    for item in qa_list:
        qid = item["id"]
        row = " | ".join(f"{r['per_q'][qid]:<10}" for r in results)
        print(f"{qid:<4} | {item['type']:<10} | {row}")

    # 自动保存结果到 CSV，方便后续写入实验日志或画图
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["chunk_size", "num_chunks", "avg_chars", "file_hit_rate", "exact_hit_rate", "avg_ctx_chars"] + [q["id"] for q in qa_list])
        for r in results:
            w.writerow([r["chunk_size"], r["num_chunks"], r["avg_chars"], r["file_hit_rate"], r["exact_hit_rate"], int(r["avg_ctx_chars"])] + [r["per_q"][q["id"]] for q in qa_list])
    print(f"\n实验结果已自动保存至：{OUT_CSV}")

import os
# 必须在导入 HuggingFace 之前设置国内镜像，防止下载中文向量模型超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # 关闭 Windows 快捷方式的无害红字提示

from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 自动定位到“当前脚本所在的 week3_rag 文件夹”，在 PyCharm 任意位置点运行都不会找不到路径
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"             # 旋钮 1：语料文件夹（换别的语料改这里）
STORAGE_DIR = BASE_DIR / "storage"       # 旋钮 2：索引存盘目录（做多组对照时改成 storage_256 等）
MODEL_DIR = BASE_DIR / "models" / "bge-small-zh-v1.5"  # 本地模型路径，完全离线加载不卡网络

# 旋钮 3：中文向量模型（优先读本地下载好的目录，秒加载；想换别的模型改这里）
Settings.embed_model = HuggingFaceEmbedding(model_name=str(MODEL_DIR))

# 旋钮 4：切片参数（周六做 256 / 512 / 1024 对比实验就改这里！）
Settings.node_parser = SentenceSplitter(
    chunk_size=512,      # 每块大约多少 token
    chunk_overlap=50,    # 相邻两块重叠多少 token，防止条款从中间被拦腰切断
)

# 1. 读取 docs/ 里的全部政策原文（支持 .txt 和 .pdf）
docs = SimpleDirectoryReader(str(DOCS_DIR)).load_data()
print(f"已加载原始文档数：{len(docs)}")

# 2. 显式切分文档块（chunk），方便核对切出了多少块、每块长什么样
nodes = Settings.node_parser.get_nodes_from_documents(docs)
print(f"切分后的文档块（chunk）总数：{len(nodes)}")
print("--- 第 1 个 chunk 预览（前 150 字） ---")
print(nodes[0].text[:150])
print("---------------------------------------")

# 3. 把切好的文本块转成向量，并保存到硬盘（明天周四直接读硬盘，不用重算）
index = VectorStoreIndex(nodes)
index.storage_context.persist(persist_dir=str(STORAGE_DIR))
print(f"向量索引已保存到：{STORAGE_DIR}")
9
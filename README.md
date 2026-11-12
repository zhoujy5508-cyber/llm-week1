# LLM 学习记录 · 第 1 周

跟着 LLM 应用开发路线做的第一周练习：打通 API 调用、写分类器、做多轮对话。

## 文件说明

| 文件 | 作用 |
|---|---|
| test_key.py | 检查 API Key 是否可用 |
| chat.py | 单轮问答 |
| chat_loop.py | 多轮对话，验证上下文记忆 |
| news_classifier.py | 矿产能源新闻分类（zero-shot vs few-shot 对比） |

## 运行方式

1. `pip install -r requirements.txt`
2. 在根目录新建 `.env`，写入 `DEEPSEEK_API_KEY=你的key`
3. `python chat.py`

## 环境

Python 3.12 / openai 3.x / DeepSeek API

## 已知局限

- 没有历史持久化，关掉程序记忆就没了
- 回答里的数字来自模型参数，不保证时效性

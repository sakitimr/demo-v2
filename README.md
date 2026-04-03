# 🌟 古诗词生成助手 (Poetry Assistant)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/MindSpore-2.0+-purple.svg" alt="MindSpore">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Model-DeepSeek--R1--Distill--Qwen--1.5B-orange.svg" alt="Model">
</p>

> 🤖 基于大语言模型的智能古诗词生成系统 | AI-Powered Classical Chinese Poetry Generator

## 📖 项目简介

古诗词助手是一个基于**深度学习**的智能古诗词生成项目，采用了当前前沿的**大语言模型 (LLM)** + **LoRA 微调**技术架构。通过对海量古诗词数据的学习，系统能够根据用户输入的主题、意象或关键词，自动创作出符合古典诗词格律的原创作品。

本项目使用 **华为 MindSpore** 框架作为深度学习后端，结合 **Transformers** 库和 **PEFT** 高效微调技术，在消费级硬件上即可完成模型训练与推理部署。

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        技术栈总览                               │
├─────────────────────────────────────────────────────────────────┤
│  🧠 模型层    │  DeepSeek-R1-Distill-Qwen-1.5B (基座模型)        │
│              │  + LoRA (Low-Rank Adaptation) 微调                │
├──────────────┼────────────────────────────────────────────────┤
│  🔧 框架层    │  MindSpore 2.0+ (华为昇腾深度学习框架)           │
│              │  Transformers 4.30+ (Hugging Face)               │
│              │  PEFT (Parameter-Efficient Fine-Tuning)         │
├──────────────┼────────────────────────────────────────────────┤
│  📊 数据层    │  古诗词语料库 (全唐诗/全宋词/全元曲等)           │
│              │  繁简转换 (OpenCC)                               │
│              │  JSON 格式训练数据                               │
├──────────────┼────────────────────────────────────────────────┤
│  🎨 应用层    │  Jupyter Notebook (交互式开发)                  │
│              │  (可扩展至 FastAPI / Gradio Web UI)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 项目结构

```
demo/
├── 📄 README.md                 # 项目说明文档
├── 📄 readme.txt                # 快速入门指南
├── 📓 数据处理.ipynb            # 古诗词数据预处理
│   ├── 古诗词数据清洗与格式化
│   ├── 繁简转换 (OpenCC)
│   ├── 训练数据 JSON 构建
│   └── tokenizer 词表构建
│
├── 📓 模型训练及处理.ipynb       # 模型训练与推理
│   ├── DeepSeek-R1 基座模型加载
│   ├── LoRA 微调配置
│   ├── 模型训练 (3 Epochs)
│   ├── 推理测试
│   └── Checkpoint 保存
│
└── 📁 checkpoints/               # 预训练模型权重
    └── checkpoint-1000/        # 已训练好的 LoRA 权重
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/sakitimr/demo.git
cd demo

# 创建虚拟环境
conda create -n poetry python=3.10
conda activate poetry

# 安装依赖
pip install mindspore
pip install transformers
pip install peft
pip install opencc-python-reimplemented
pip install pandas jieba datasets
```

### 2. 数据处理

```bash
# 运行数据处理 Notebook
jupyter notebook 数据处理.ipynb
```

### 3. 模型训练

```bash
# 确保已下载 DeepSeek-R1-Distill-Qwen-1.5B 模型
# 运行训练 Notebook
jupyter notebook 模型训练及处理.ipynb
```

### 4. 推理测试

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基座模型
model_path = "./DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval()

# 加载 LoRA 权重
lora_path = "./checkpoint-1000"
model = PeftModel.from_pretrained(model, model_id=lora_path)

# 生成古诗
prompt = "写一首关于春天的七言绝句"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=128)
poem = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(poem)
```

---

## 🎯 功能特性

| 功能 | 说明 |
|------|------|
| ✍️ **诗词创作** | 根据主题/关键词自动生成五言/七言绝句、律诗、宋词 |
| 🎨 **风格迁移** | 可学习李白、杜甫、苏轼等不同诗人的风格 |
| 📊 **数据分析** | 内置古诗词词频统计、词云生成等功能 |
| 🔄 **繁简转换** | 支持繁体与简体中文互转 |
| 📦 **模型压缩** | 支持 LoRA 权重导出与模型量化 |

---

## 📈 训练细节

| 参数 | 值 |
|------|------|
| 基座模型 | DeepSeek-R1-Distill-Qwen-1.5B |
| 微调方法 | LoRA (r=8, alpha=32) |
| 训练轮次 | 3 Epochs |
| 批次大小 | 4 |
| 梯度累积 | 5 |
| 学习率 | 1e-4 |
| 最大序列长度 | 384 |

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/xxx`)
3. 提交更改 (`git commit -m 'Add xxx'`)
4. 推送分支 (`git push origin feature/xxx`)
5. 打开 Pull Request

---

## 📄 许可证

MIT License - 自由使用，署名引用

---

## 🙏 致谢

- [华为 MindSpore](https://www.mindspore.cn/) - 深度学习框架
- [Hugging Face Transformers](https://huggingface.co/) - 模型库
- [DeepSeek](https://github.com/deepseek-ai) - 基座模型
- [全唐诗/全宋词](https://github.com/) - 古诗词语料

---

## 📧 联系方式

如有问题，欢迎提交 Issue 或联系维护者。

---

<p align="center">
  <sub>Made with ❤️ by AI Poetry Team</sub>
</p>
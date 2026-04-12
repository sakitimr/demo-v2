# ============================================================
# 古诗词生成助手 - Gradio 交互应用
# Poetry-Assistant / Gradio Web UI
# ============================================================
# 部署方式：
#   本地运行：python app.py
#   HuggingFace Spaces：推送到 spaces 即可自动部署
#   （需要先上传 checkpoints/ 到仓库或 HuggingFace Hub）
# ============================================================

import gradio as gr
import os
import re

# -----------------------------------------------
# 模型加载（延迟加载，首次推理时才初始化）
# -----------------------------------------------
_model = None
_tokenizer = None
_lora_loaded = False

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)
LORA_PATH = os.environ.get(
    "LORA_PATH",
    "sakitimr/Poetry-Assistant/ checkpoint-1000"
)

def load_model():
    """延迟加载模型，避免 Space 冷启动超时"""
    global _model, _tokenizer, _lora_loaded
    if _model is not None:
        return

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        gr.Info("正在加载基座模型，请稍候（首次可能需要 3-5 分钟）...")
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        )
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        ).eval()

        # 尝试加载 LoRA 权重
        try:
            _model = PeftModel.from_pretrained(_model, LORA_PATH)
            _lora_loaded = True
            gr.Info("✓ LoRA 权重加载成功")
        except Exception:
            _lora_loaded = False
            gr.Warning("LoRA 权重未找到，将使用基座模型推理")

        gr.Info("✓ 模型加载完成")
    except Exception as e:
        gr.Error(f"模型加载失败: {e}")


def generate_poetry(
    theme: str,
    poetry_type: str,
    poet_style: str,
    temperature: float,
    max_length: int
) -> str:
    """
    根据用户输入生成古诗词

    Args:
        theme:       主题/意象/关键词
        poetry_type: 诗体类型
        poet_style:  诗人风格
        temperature: 随机性控制
        max_length:  最大生成长度
    """
    if not theme or not theme.strip():
        return "⚠️ 请输入诗词主题或关键词"

    load_model()

    # 构建 Prompt
    type_hint = {
        "七言绝句": "七言绝句",
        "五言绝句": "五言绝句",
        "七言律诗": "七言律诗",
        "五言律诗": "五言律诗",
        "宋词·短调": "宋词（短调）",
        "宋词·长调": "宋词（长调）",
        "元曲": "元曲",
        "不限": "",
    }.get(poetry_type, "")

    style_hint = {
        "李白（豪放飘逸）": "李白风格，豪放飘逸，想象奇特",
        "杜甫（沉郁顿挫）": "杜甫风格，沉郁顿挫，家国情怀",
        "苏轼（旷达洒脱）": "苏轼风格，旷达洒脱，哲理深刻",
        "李清照（婉约清丽）": "李清照风格，婉约清丽，情感细腻",
        "不限": "",
    }.get(poet_style, "")

    # 构造中文 Prompt（Instruct 格式）
    parts = [f"请以「{theme.strip()}」为题"]
    if type_hint:
        parts.append(f"写一首{type_hint}")
    else:
        parts.append("写一首古诗词")
    if style_hint:
        parts.append(f"，模仿{style_hint}")
    parts.append("\n\n请直接输出诗词正文，不要解释。")
    prompt = "".join(parts)

    # 生成
    try:
        inputs = _tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=_tokenizer.eos_token_id,
        )

        generated = _tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )

        # 清理思考链（DeepSeek R1 的 think 标签）
        generated = re.sub(
            r'<[^>]+>', '', generated
        ).strip()

        if not generated:
            return "⚠️ 生成内容为空，请尝试调整参数或更换主题"

        return generated

    except Exception as e:
        return f"⚠️ 生成失败: {e}"


# -----------------------------------------------
# 示例输入
# -----------------------------------------------
EXAMPLES = [
    ["春天", "七言绝句", "不限", 0.7, 128],
    ["月亮、思乡", "五言律诗", "杜甫（沉郁顿挫）", 0.8, 192],
    ["边塞、战争", "七言律诗", "李白（豪放飘逸）", 0.75, 192],
    ["秋日黄昏", "宋词·短调", "苏轼（旷达洒脱）", 0.8, 160],
    ["雨夜、孤独", "宋词·长调", "李清照（婉约清丽）", 0.7, 256],
    ["松柏、坚韧", "五言绝句", "不限", 0.6, 128],
]


# -----------------------------------------------
# 构建 Gradio 界面
# -----------------------------------------------
with gr.Blocks(
    title="古诗词生成助手",
    theme=gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="stone",
        neutral_hue="stone",
        font=["Noto Serif SC", gr.themes.GoogleFont("Noto Serif SC")],
    ),
    css="""
    /* 水墨风主题定制 */
    .gradio-container { background: #0d1117; }
    .poetry-output { font-family: 'Noto Serif SC', serif !important; }
    #result-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e) !important;
        border: 1px solid rgba(201,168,76,0.3) !important;
        border-radius: 8px !important;
        color: #e8dfc8 !important;
        font-size: 1.1rem !important;
        line-height: 2.2 !important;
        padding: 24px !important;
        white-space: pre-line !important;
        letter-spacing: 2px;
    }
    """,
) as demo:

    gr.Markdown(
        """
        <div style="text-align:center; padding: 20px 0;">
            <h1 style="
                font-family: 'Noto Serif SC', serif;
                color: #f0e8d0;
                font-size: 2.2rem;
                letter-spacing: 6px;
                margin-bottom: 8px;
            ">✦ 古诗词生成助手 ✦</h1>
            <p style="color:#8a7a60; font-size:0.9rem; letter-spacing:2px;">
                基于 DeepSeek-R1 + MindSpore + LoRA 微调 | AI-Powered Classical Chinese Poetry
            </p>
        </div>
        """,
        elem_id="header",
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            theme_input = gr.Textbox(
                label="📝 诗词主题 / 意象",
                placeholder="例如：春天、明月、思乡、边塞秋风、雨夜独酌……",
                lines=2,
                value="",
            )

            with gr.Row():
                type_input = gr.Dropdown(
                    label="📜 诗体",
                    choices=[
                        "七言绝句", "五言绝句",
                        "七言律诗", "五言律诗",
                        "宋词·短调", "宋词·长调",
                        "元曲", "不限"
                    ],
                    value="七言绝句",
                )
                style_input = gr.Dropdown(
                    label="🎭 诗人风格",
                    choices=[
                        "不限", "李白（豪放飘逸）",
                        "杜甫（沉郁顿挫）", "苏轼（旷达洒脱）",
                        "李清照（婉约清丽）",
                    ],
                    value="不限",
                )

            with gr.Row():
                temp_input = gr.Slider(
                    label="🌡 随机性",
                    minimum=0.3, maximum=1.2, step=0.05, value=0.7,
                    info="越大越有创意，越小越稳定",
                )
                length_input = gr.Slider(
                    label="📏 生成长度",
                    minimum=64, maximum=384, step=16, value=128,
                    info="最大 token 数",
                )

            generate_btn = gr.Button(
                "✨ 生成诗词",
                variant="primary",
                size="lg",
            )

        with gr.Column(scale=1):
            result_output = gr.Textbox(
                label="🎋 生成结果",
                lines=12,
                interactive=False,
                elem_id="result-box",
            )
            gr.Examples(
                examples=EXAMPLES,
                inputs=[theme_input, type_input, style_input, temp_input, length_input],
                label="💡 示例主题",
            )

    generate_btn.click(
        fn=generate_poetry,
        inputs=[theme_input, type_input, style_input, temp_input, length_input],
        outputs=result_output,
    )
    theme_input.submit(
        fn=generate_poetry,
        inputs=[theme_input, type_input, style_input, temp_input, length_input],
        outputs=result_output,
    )

    gr.Markdown(
        """
        <div style="
            text-align:center; padding: 20px;
            border-top: 1px solid rgba(201,168,76,0.1);
            margin-top: 20px;
            color:#5a4a30; font-size:0.78rem; letter-spacing:1px;
        ">
            Poetry-Assistant ·
            <a href="https://github.com/sakitimr/Poetry-Assistant" target="_blank"
               style="color:#8a7040; text-decoration:none;">
                GitHub @sakitimr
            </a>
            · MIT License
        </div>
        """,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )

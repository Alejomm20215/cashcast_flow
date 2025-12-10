"""
Utility to generate a Kaggle-ready notebook for a small LoRA fine-tune on a 7B model.
The notebook metadata requests a GPU accelerator (T4 on Kaggle when available).

Run:
  python scripts/generate_kaggle_lora_notebook.py

It will write: kaggle_lora_finetune.ipynb
Upload that notebook to Kaggle, open the right sidebar, and select GPU.
"""

import nbformat as nbf


def main():
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        # Kaggle honors this as a request; you still need to select GPU in the sidebar if available.
        "accelerator": "GPU",
    }

    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# LoRA Fine-tune (Kaggle)\n"
            "- Requests GPU (T4/P100) via notebook metadata; select GPU in the sidebar.\n"
            "- Keeps batch small to fit free GPUs.\n"
            "- Replace `train.jsonl` with your data (prompt/response pairs).\n"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "!pip install -q transformers peft datasets accelerate bitsandbytes huggingface_hub"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "import json, torch, os\n"
            "from datasets import load_dataset\n"
            "from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer\n"
            "from peft import LoraConfig, get_peft_model\n"
            "from huggingface_hub import login\n\n"
            "# --- Config ---\n"
            "model_name = os.getenv('BASE_MODEL', 'mistralai/Mistral-7B-Instruct-v0.2')\n"
            "train_file = os.getenv('TRAIN_FILE', 'train.jsonl')  # upload a JSONL with prompt/response\n"
            "output_dir = 'lora_out'\n"
            "use_8bit = True  # for T4/P100 memory fit\n"
            "hf_token = os.getenv('HF_TOKEN')  # set in Kaggle Secrets or env\n\n"
            "if hf_token:\n"
            "    login(token=hf_token)\n"
            "else:\n"
            "    print('Warning: HF_TOKEN not set; cannot push to Hub unless you login later.')\n\n"
            "tokenizer = AutoTokenizer.from_pretrained(model_name)\n"
            "model = AutoModelForCausalLM.from_pretrained(\n"
            "    model_name,\n"
            "    load_in_8bit=use_8bit,\n"
            "    device_map='auto',\n"
            ")\n\n"
            "lora_cfg = LoraConfig(\n"
            "    r=8,\n"
            "    lora_alpha=16,\n"
            "    lora_dropout=0.05,\n"
            "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\"],\n"
            "    bias=\"none\",\n"
            "    task_type=\"CAUSAL_LM\",\n"
            ")\n"
            "model = get_peft_model(model, lora_cfg)\n\n"
            "def format_example(ex):\n"
            "    text = f\"<s>{ex['prompt']}\\n{ex['response']}</s>\"\n"
            "    return tokenizer(text, truncation=True, max_length=512)\n\n"
            "ds = load_dataset('json', data_files=train_file)['train'].map(format_example, remove_columns=['prompt','response'])\n\n"
            "args = TrainingArguments(\n"
            "    output_dir=output_dir,\n"
            "    per_device_train_batch_size=1,\n"
            "    gradient_accumulation_steps=4,\n"
            "    num_train_epochs=1,\n"
            "    learning_rate=2e-4,\n"
            "    fp16=True,\n"
            "    logging_steps=10,\n"
            "    save_total_limit=1,\n"
            ")\n\n"
            "trainer = Trainer(model=model, args=args, train_dataset=ds)\n"
            "trainer.train()\n\n"
            "model.save_pretrained(output_dir)\n"
            "tokenizer.save_pretrained(output_dir)\n"
            "print('Saved LoRA adapter to', output_dir)\n"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# Optional: push to HF Hub (requires HF_TOKEN)\n"
            "!huggingface-cli upload lora_out your-username/your-lora-repo -r main --include '*'"
        )
    )

    nb["cells"] = cells
    with open("kaggle_lora_finetune.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote kaggle_lora_finetune.ipynb (requests GPU).")


if __name__ == "__main__":
    main()


"""LoRA fine-tune a small open model on the SFT files (TRL + PEFT).
   pip install -r requirements-train.txt
   python -m train.train_lora --data data/sft --out out/dm-lora
   Untested on GPU in this repo's CI: TRL's API shifts between versions, so check your installed version."""
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data/sft")
    ap.add_argument("--out", default="out/dm-lora")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    a = ap.parse_args()

    ds = load_dataset("json", data_files={"train": f"{a.data}/train.jsonl", "validation": f"{a.data}/val.jsonl"})
    lora = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
                    per_device_train_batch_size=2, gradient_accumulation_steps=8,
                    logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
                    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported())  # T4 (free Colab) has no bf16
    trainer = SFTTrainer(model=a.base, args=cfg, train_dataset=ds["train"],
                         eval_dataset=ds["validation"], peft_config=lora)
    trainer.train()
    trainer.save_model(a.out)
    print("adapter saved to", a.out)
    print("serve:  vllm serve", a.base, "--enable-lora --lora-modules dm=" + a.out,
          "--enable-auto-tool-choice --tool-call-parser hermes")
    print("play:   DM_BACKEND=openai DM_MODEL=dm DM_OPENAI_BASE_URL=http://localhost:8000/v1 python -m app.cli")


if __name__ == "__main__":
    main()

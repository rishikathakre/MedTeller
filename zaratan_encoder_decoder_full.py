# -*- coding: utf-8 -*-
"""
Encoder-Decoder Integration for Zaratan HPC
Includes: Training, Evaluation, Metrics (BLEU/ROUGE), Logging, and Plots
"""

import os
import sys
import math
import json
import time
from datetime import datetime
from collections import defaultdict

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION - Paths match your Zaratan scratch setup
# ============================================================================
CONFIG = {
    # Base directory (set by --chdir in slurm)
    "data_dir": "data",
    "output_dir": "outputs",
    
    # Model paths (your pretrained models from Google Drive)
    "decoder_tokenizer_dir": "data/decoder_tokenizer",
    "decoder_pretrained_dir": "data/decoder_pretrained",
    "multimodal_decoder_dir": "data/full_multimodal_decoder",
    
    # Mode: "train", "evaluate", or "both"
    "mode": "both",
    
    # Subsample for quick testing (set to None for full data)
    "max_train_samples": None,
    "max_val_samples": None,
    
    # Multimodal decoder hyperparams (optimized for A100)
    "num_epochs": 15,
    "learning_rate": 3e-5,
    "train_batch_size": 8,         # A100 can handle larger batches
    "eval_batch_size": 16,         # Faster evaluation
    "gradient_accumulation": 2,    # Effective batch = 16
    "max_length": 256,
    "cond_dim": 1536,  # 768 (ViT pooled) + 768 (ClinicalBERT)
    
    # Logging (adjusted for larger batch size)
    "logging_steps": 20,
    "save_steps": 250,
    "eval_steps": 250,
    
    # Evaluation settings
    "num_generate_samples": 20,
    "max_new_tokens": 150,
    
    # Seed
    "seed": 42,
}


# ============================================================================
# SETUP
# ============================================================================
def setup_environment():
    """Set up the environment and check GPU availability."""
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("ENCODER-DECODER INTEGRATION - ZARATAN")
    print("=" * 70)
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"\nMode: {CONFIG['mode']}")
    print("=" * 70)
    
    # Create output directories
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "plots"), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["output_dir"], "generations"), exist_ok=True)
    
    return device


def load_multimodal_data():
    """Load the multimodal dataset from .pt file."""
    mm_path = os.path.join(CONFIG["data_dir"], "multimodal_dataset_full.pt")
    
    if not os.path.exists(mm_path):
        mm_path = os.path.join(CONFIG["data_dir"], "multimodal_dataset.pt")
    
    if not os.path.exists(mm_path):
        raise FileNotFoundError(
            f"Could not find multimodal dataset. "
            f"Please ensure multimodal_dataset_full.pt exists in {CONFIG['data_dir']}"
        )
    
    print(f"\nLoading multimodal dataset from: {mm_path}")
    mm_data = torch.load(mm_path, map_location="cpu")
    
    print(f"Dataset splits: {mm_data.keys()}")
    if "metadata" in mm_data:
        print(f"Metadata: {mm_data.get('metadata', {})}")
    
    if "train" in mm_data and len(mm_data["train"]) > 0:
        sample = mm_data["train"][0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Image embedding shape: {sample['image_emb'].shape}")
        print(f"Text embedding shape: {sample['text_emb'].shape}")
    
    return mm_data


# ============================================================================
# DATASET CLASS
# ============================================================================
class MultimodalReportDataset(Dataset):
    """Dataset for multimodal report generation."""
    
    def __init__(self, samples, tokenizer, max_length=256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        text = s.get("full_report", "") or s.get("impression", "") or ""
        text = str(text)
        
        prompt = (self.tokenizer.bos_token + " " + text).strip()
        
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        labels = input_ids.clone()
        
        img_emb = s["image_emb"]
        txt_emb = s["text_emb"]
        
        if img_emb.dim() == 2:
            img_pooled = img_emb.mean(dim=0)
        else:
            img_pooled = img_emb
        
        cond_vec = torch.cat([img_pooled, txt_emb], dim=-1)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "cond_vec": cond_vec,
        }


def multimodal_collate_fn(batch, pad_token_id):
    """Collate function for multimodal batches."""
    input_ids = [b["input_ids"] for b in batch]
    attention_masks = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]
    cond_vecs = [b["cond_vec"] for b in batch]
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    cond_vecs = torch.stack(cond_vecs, dim=0)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "cond_vec": cond_vecs,
    }


# ============================================================================
# MODEL CLASS
# ============================================================================
class GPT2WithConditioning(nn.Module):
    """GPT-2 wrapper with encoder conditioning prefix."""
    
    def __init__(self, base_model, cond_dim=1536):
        super().__init__()
        self.gpt2 = base_model
        self.cond_proj = nn.Linear(cond_dim, self.gpt2.config.n_embd)
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, cond_vec=None, **kwargs):
        cond_emb = self.cond_proj(cond_vec)
        cond_emb = cond_emb.unsqueeze(1)
        
        token_emb = self.gpt2.transformer.wte(input_ids)
        inputs_embeds = torch.cat([cond_emb, token_emb], dim=1)
        
        if attention_mask is not None:
            prefix_mask = torch.ones(
                (attention_mask.size(0), 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        if labels is not None:
            prefix_labels = -100 * torch.ones(
                (labels.size(0), 1),
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([prefix_labels, labels], dim=1)
        
        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs


# ============================================================================
# GENERATION FUNCTION
# ============================================================================
def generate_from_sample(model, tokenizer, sample, device, max_new_tokens=150):
    """Generate text from a single sample."""
    model.eval()
    with torch.no_grad():
        img_emb = sample["image_emb"]
        txt_emb = sample["text_emb"]
        
        if img_emb.dim() == 2:
            img_pooled = img_emb.mean(dim=0)
        else:
            img_pooled = img_emb
        
        cond_vec = torch.cat([img_pooled, txt_emb], dim=-1).unsqueeze(0).to(device)
        
        generated_ids = tokenizer(
            tokenizer.bos_token,
            return_tensors="pt"
        )["input_ids"].to(device)
        
        for _ in range(max_new_tokens):
            token_emb = model.gpt2.transformer.wte(generated_ids)
            cond_emb = model.cond_proj(cond_vec).unsqueeze(1)
            inputs_embeds = torch.cat([cond_emb, token_emb], dim=1)
            
            attn_mask = torch.ones(
                inputs_embeds.size()[:2],
                dtype=torch.long,
                device=device,
            )
            
            outputs = model.gpt2(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_id], dim=1)
            
            if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
                break
        
        gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return gen_text


# ============================================================================
# EVALUATION METRICS
# ============================================================================
def compute_metrics(predictions, references):
    """Compute BLEU and ROUGE metrics."""
    import re
    from collections import Counter
    
    def tokenize(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def bleu_score(pred, ref, max_n=4):
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)
        
        if len(pred_tokens) == 0:
            return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)}
        
        scores = {}
        for n in range(1, max_n + 1):
            pred_ngrams = ngrams(pred_tokens, n)
            ref_ngrams = ngrams(ref_tokens, n)
            
            if len(pred_ngrams) == 0:
                scores[f"bleu_{n}"] = 0.0
                continue
            
            pred_counts = Counter(pred_ngrams)
            ref_counts = Counter(ref_ngrams)
            
            matches = sum(min(pred_counts[ng], ref_counts[ng]) for ng in pred_counts)
            precision = matches / len(pred_ngrams)
            scores[f"bleu_{n}"] = precision
        
        return scores
    
    def rouge_l(pred, ref):
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        m, n = len(pred_tokens), len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i-1] == ref_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_len = dp[m][n]
        precision = lcs_len / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    all_bleu = defaultdict(list)
    all_rouge = []
    
    for pred, ref in zip(predictions, references):
        bleu = bleu_score(pred, ref)
        for k, v in bleu.items():
            all_bleu[k].append(v)
        all_rouge.append(rouge_l(pred, ref))
    
    metrics = {}
    for k, v in all_bleu.items():
        metrics[k] = np.mean(v)
    metrics["rouge_l"] = np.mean(all_rouge)
    
    return metrics


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def plot_training_curves(log_history, output_dir):
    """Plot training and validation loss curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("step", len(train_steps)))
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", len(eval_steps)))
            eval_losses.append(entry["eval_loss"])
    
    if not train_losses:
        print("No training losses to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(train_steps, train_losses, 'b-', alpha=0.7, label='Train Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if eval_losses:
        axes[1].plot(eval_steps, eval_losses, 'r-o', label='Eval Loss')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Validation Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "training_curves.png"), dpi=150)
    plt.close()
    print(f"Saved training curves to: {output_dir}/plots/training_curves.png")


def plot_metrics_bar(metrics, output_dir):
    """Plot evaluation metrics as bar chart."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(metrics.keys())
    values = list(metrics.values())
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    
    bars = ax.bar(names, values, color=colors[:len(names)])
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Score')
    ax.set_title('Evaluation Metrics')
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "metrics.png"), dpi=150)
    plt.close()
    print(f"Saved metrics plot to: {output_dir}/plots/metrics.png")


# ============================================================================
# LOAD MODELS
# ============================================================================
def load_models(device):
    """Load pretrained tokenizer and decoder from Google Drive uploads."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast
    
    print("\n" + "=" * 70)
    print("LOADING PRETRAINED MODELS")
    print("=" * 70)
    
    tokenizer_dir = CONFIG["decoder_tokenizer_dir"]
    decoder_dir = CONFIG["decoder_pretrained_dir"]
    
    # Verify pretrained models exist
    if not os.path.exists(decoder_dir):
        raise FileNotFoundError(
            f"Decoder not found at: {decoder_dir}\n"
            f"Please upload decoder_pretrained/ folder from Google Drive."
        )
    
    # Load tokenizer with fallback for version mismatch
    tokenizer = None
    
    if os.path.exists(tokenizer_dir):
        print(f"Loading tokenizer from: {tokenizer_dir}")
        
        # Try 1: Normal loading
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            print("  Loaded tokenizer successfully")
        except Exception as e1:
            print(f"  Warning: Normal load failed: {e1}")
            
            # Try 2: Load with use_fast=False (slow tokenizer)
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)
                print("  Loaded slow tokenizer successfully")
            except Exception as e2:
                print(f"  Warning: Slow tokenizer load failed: {e2}")
    
    # Fallback: Recreate tokenizer from base GPT-2 with same special tokens
    if tokenizer is None:
        print("  Recreating tokenizer from base GPT-2 with medical special tokens...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Add special tokens (same as used in training)
        special_tokens = {
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "pad_token": "<PAD>",
            "additional_special_tokens": ["<FINDINGS>", "<IMPRESSION>"],
        }
        tokenizer.add_special_tokens(special_tokens)
        
        # Add medical tokens
        med_tokens = [
            "cardiomegaly", "atelectasis", "consolidation", "effusion", "pneumothorax",
            "edema", "collapse", "opacity", "opacities", "hyperinflation", "fibrosis",
            "infiltrate", "infiltrates", "pleural", "interstitial"
        ]
        tokenizer.add_tokens(med_tokens)
        print(f"  Recreated tokenizer with vocab size: {len(tokenizer)}")
    
    print(f"Loading decoder from: {decoder_dir}")
    model_base = AutoModelForCausalLM.from_pretrained(decoder_dir)
    
    # Ensure model embeddings match tokenizer vocab size
    if model_base.config.vocab_size != len(tokenizer):
        print(f"  Resizing model embeddings from {model_base.config.vocab_size} to {len(tokenizer)}")
        model_base.resize_token_embeddings(len(tokenizer))
    
    model_base = model_base.to(device)
    
    print(f"Tokenizer - bos: {tokenizer.bos_token}, eos: {tokenizer.eos_token}, pad: {tokenizer.pad_token}")
    print(f"GPT-2 hidden size: {model_base.config.n_embd}")
    
    # Create multimodal model
    mm_model = GPT2WithConditioning(model_base, cond_dim=CONFIG["cond_dim"]).to(device)
    mm_model.gpt2.config.use_cache = False
    
    # Untie shared weights
    with torch.no_grad():
        mm_model.gpt2.lm_head.weight = nn.Parameter(
            mm_model.gpt2.lm_head.weight.clone()
        )
    
    # Load pretrained multimodal weights if available
    mm_state_path = os.path.join(CONFIG["multimodal_decoder_dir"], "mm_model_state.pt")
    if os.path.exists(mm_state_path):
        print(f"\nLoading pretrained multimodal weights from: {mm_state_path}")
        state_dict = torch.load(mm_state_path, map_location=device)
        mm_model.load_state_dict(state_dict)
        print("Loaded pretrained multimodal model!")
    else:
        print("\nNo pretrained multimodal weights found, will train from scratch")
    
    print(f"\nTotal model parameters: {sum(p.numel() for p in mm_model.parameters()):,}")
    
    return tokenizer, mm_model


# ============================================================================
# TRAINING
# ============================================================================
def train_model(mm_model, mm_data, tokenizer, device):
    """Train the multimodal decoder."""
    from transformers import TrainingArguments, Trainer
    from functools import partial
    
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    train_samples = mm_data.get("train", [])
    val_samples = mm_data.get("val", [])
    
    if CONFIG["max_train_samples"]:
        train_samples = train_samples[:CONFIG["max_train_samples"]]
    if CONFIG["max_val_samples"]:
        val_samples = val_samples[:CONFIG["max_val_samples"]]
    
    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    mm_train_ds = MultimodalReportDataset(train_samples, tokenizer, max_length=CONFIG["max_length"])
    mm_val_ds = MultimodalReportDataset(val_samples, tokenizer, max_length=CONFIG["max_length"])
    
    output_dir = os.path.join(CONFIG["output_dir"], "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["train_batch_size"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=150,
        weight_decay=0.01,
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        eval_steps=CONFIG["eval_steps"],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=6,
    )
    
    collate_fn = partial(multimodal_collate_fn, pad_token_id=tokenizer.pad_token_id)
    
    trainer = Trainer(
        model=mm_model,
        args=training_args,
        train_dataset=mm_train_ds,
        eval_dataset=mm_val_ds,
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )
    
    print("\nStarting training...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time/60:.2f} minutes")
    
    # Evaluate
    eval_results = trainer.evaluate()
    print(f"\nEval results: {eval_results}")
    
    if "eval_loss" in eval_results:
        val_ppl = math.exp(eval_results["eval_loss"])
        print(f"Validation Perplexity: {val_ppl:.4f}")
    
    # Save model
    save_dir = os.path.join(CONFIG["output_dir"], "final_model")
    os.makedirs(save_dir, exist_ok=True)
    
    mm_model.gpt2.save_pretrained(os.path.join(save_dir, "gpt2"))
    tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
    torch.save(mm_model.state_dict(), os.path.join(save_dir, "mm_model_state.pt"))
    
    print(f"\nSaved model to: {save_dir}")
    
    # Plot training curves
    plot_training_curves(trainer.state.log_history, CONFIG["output_dir"])
    
    # Save log history
    log_path = os.path.join(CONFIG["output_dir"], "training_log.json")
    with open(log_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    
    return trainer, eval_results


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_model(mm_model, mm_data, tokenizer, device):
    """Evaluate the model and compute metrics."""
    print("\n" + "=" * 70)
    print("EVALUATION - Generating Samples & Computing Metrics")
    print("=" * 70)
    
    val_samples = mm_data.get("val", [])
    if CONFIG["max_val_samples"]:
        val_samples = val_samples[:CONFIG["max_val_samples"]]
    
    num_samples = min(CONFIG["num_generate_samples"], len(val_samples))
    print(f"\nGenerating {num_samples} samples for evaluation...")
    
    predictions = []
    references = []
    generation_results = []
    
    mm_model.eval()
    for i in range(num_samples):
        sample = val_samples[i]
        
        gen_text = generate_from_sample(
            mm_model, tokenizer, sample, device,
            max_new_tokens=CONFIG["max_new_tokens"]
        )
        
        ref_text = sample.get("full_report", "") or sample.get("impression", "") or ""
        
        predictions.append(gen_text)
        references.append(ref_text)
        
        generation_results.append({
            "idx": i,
            "generated": gen_text,
            "reference": ref_text,
            "impression": sample.get("impression", ""),
        })
        
        # Print first 3 samples
        if i < 3:
            print(f"\n{'='*60}")
            print(f"Sample {i}")
            print(f"{'='*60}")
            print(f"GENERATED:\n{gen_text[:400]}...")
            print(f"\nREFERENCE:\n{ref_text[:400]}...")
    
    # Compute metrics
    print("\n" + "-" * 60)
    print("METRICS")
    print("-" * 60)
    
    metrics = compute_metrics(predictions, references)
    
    print("\nEvaluation Metrics:")
    print(f"  BLEU-1: {metrics['bleu_1']:.4f}")
    print(f"  BLEU-2: {metrics['bleu_2']:.4f}")
    print(f"  BLEU-3: {metrics['bleu_3']:.4f}")
    print(f"  BLEU-4: {metrics['bleu_4']:.4f}")
    print(f"  ROUGE-L: {metrics['rouge_l']:.4f}")
    
    # Plot metrics
    plot_metrics_bar(metrics, CONFIG["output_dir"])
    
    # Save results
    results_path = os.path.join(CONFIG["output_dir"], "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "num_samples": num_samples,
            "generations": generation_results,
        }, f, indent=2)
    
    print(f"\nSaved evaluation results to: {results_path}")
    
    # Save generations to text file
    gen_path = os.path.join(CONFIG["output_dir"], "generations", "samples.txt")
    with open(gen_path, "w") as f:
        for res in generation_results:
            f.write(f"{'='*60}\n")
            f.write(f"Sample {res['idx']}\n")
            f.write(f"{'='*60}\n")
            f.write(f"\nGENERATED:\n{res['generated']}\n")
            f.write(f"\nREFERENCE:\n{res['reference']}\n")
            f.write(f"\nIMPRESSION:\n{res['impression']}\n\n")
    
    print(f"Saved generations to: {gen_path}")
    
    return metrics, generation_results


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("ENCODER-DECODER INTEGRATION")
    print("=" * 70 + "\n")
    
    # Setup
    device = setup_environment()
    
    # Save config
    config_path = os.path.join(CONFIG["output_dir"], "config.json")
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    
    # Load data
    mm_data = load_multimodal_data()
    
    # Load models
    tokenizer, mm_model = load_models(device)
    
    trainer = None
    eval_results = None
    
    # Training
    if CONFIG["mode"] in ["train", "both"]:
        trainer, eval_results = train_model(mm_model, mm_data, tokenizer, device)
    
    # Evaluation
    if CONFIG["mode"] in ["evaluate", "both"]:
        metrics, generations = evaluate_model(mm_model, mm_data, tokenizer, device)
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nOutput directory: {CONFIG['output_dir']}")
    print(f"Mode: {CONFIG['mode']}")
    
    if eval_results and "eval_loss" in eval_results:
        print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
        print(f"Final perplexity: {math.exp(eval_results['eval_loss']):.4f}")
    
    print("\nDone!")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Training Script
=========================
Main training loop for the Vision-Language Transformer model.

Usage:
    python train.py --config config.json
    python train.py --epochs 50 --batch_size 16 --lr 5e-5
    
For Zaratan HPC:
    sbatch run_training.slurm
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

# Project imports
from config import Config, get_config, BASELINE_METRICS
from model import MedTeller
from dataset import ChestXrayDataset, create_dataloaders
from evaluate import Evaluator, compute_all_metrics

# Try importing wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like BLEU
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta


class TrainingMetrics:
    """Track and aggregate training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.loss_sum = 0.0
        self.num_samples = 0
        self.num_batches = 0
    
    def update(self, loss: float, batch_size: int):
        self.loss_sum += loss * batch_size
        self.num_samples += batch_size
        self.num_batches += 1
    
    @property
    def avg_loss(self) -> float:
        if self.num_samples == 0:
            return 0.0
        return self.loss_sum / self.num_samples


class Trainer:
    """Main training class for MedTeller model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        logger: logging.Logger
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        
        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=-100,  # Padding token
            label_smoothing=config.training.label_smoothing
        )
        
        # Setup mixed precision training
        self.scaler = GradScaler() if config.training.fp16 else None
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            mode='min' if 'loss' in config.training.early_stopping_metric else 'max'
        )
        
        # Setup evaluator
        self.evaluator = Evaluator(config)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_bleu4 = 0.0
        self.training_history = []
        
        # Setup directories
        os.makedirs(config.logging.checkpoint_dir, exist_ok=True)
        os.makedirs(config.logging.log_dir, exist_ok=True)
    
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters that should/shouldn't have weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.config.training.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_epsilon
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        total_steps = len(self.train_loader) * self.config.training.num_epochs
        warmup_steps = self.config.training.warmup_steps
        
        # Linear warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Cosine decay after warmup
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-7
        )
        
        # Combine schedulers
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
        
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics = TrainingMetrics()
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.num_epochs}",
            leave=True
        )
        
        accumulation_steps = self.config.training.gradient_accumulation_steps
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if self.config.training.fp16 and self.scaler is not None:
                with autocast():
                    outputs = self.model(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs['loss'] / accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss'] / accumulation_steps
                loss.backward()
            
            # Update metrics
            metrics.update(loss.item() * accumulation_steps, images.size(0))
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.training.logging_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.logger.info(
                        f"Step {self.global_step} | Loss: {metrics.avg_loss:.4f} | LR: {current_lr:.2e}"
                    )
                    
                    if WANDB_AVAILABLE and self.config.logging.use_wandb:
                        wandb.log({
                            'train/loss': metrics.avg_loss,
                            'train/learning_rate': current_lr,
                            'train/step': self.global_step
                        })
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{metrics.avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        return {'train_loss': metrics.avg_loss}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model on validation set."""
        self.model.eval()
        metrics = TrainingMetrics()
        
        all_predictions = []
        all_references = []
        
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        for batch in progress_bar:
            # Move batch to device
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass for loss
            outputs = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            metrics.update(outputs['loss'].item(), images.size(0))
            
            # Generate predictions for metric computation
            generated_ids = self.model.generate(
                images=images,
                max_length=self.config.evaluation.max_generate_length,
                num_beams=self.config.evaluation.num_beams,
                do_sample=self.config.evaluation.do_sample
            )
            
            # Decode predictions and references
            predictions = self.model.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            
            # Replace -100 (ignore index) with pad token before decoding
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = self.model.tokenizer.pad_token_id or 0
            references = self.model.tokenizer.batch_decode(
                labels_for_decode, skip_special_tokens=True
            )
            
            all_predictions.extend(predictions)
            all_references.extend(references)
            
            progress_bar.set_postfix({'loss': f'{metrics.avg_loss:.4f}'})
        
        # Compute evaluation metrics
        eval_metrics = self.evaluator.compute_metrics(all_predictions, all_references)
        eval_metrics['val_loss'] = metrics.avg_loss
        
        return eval_metrics
    
    def train(self):
        """Main training loop."""
        self.logger.info("=" * 60)
        self.logger.info("Starting Training")
        self.logger.info("=" * 60)
        self.logger.info(f"Total epochs: {self.config.training.num_epochs}")
        self.logger.info(f"Batch size: {self.config.training.batch_size}")
        self.logger.info(f"Gradient accumulation: {self.config.training.gradient_accumulation_steps}")
        self.logger.info(f"Effective batch size: {self.config.training.batch_size * self.config.training.gradient_accumulation_steps}")
        self.logger.info(f"Learning rate: {self.config.training.learning_rate}")
        self.logger.info(f"Warmup steps: {self.config.training.warmup_steps}")
        self.logger.info(f"Success threshold (BLEU-4): {BASELINE_METRICS['success_threshold']['bleu_4']}")
        self.logger.info("=" * 60)
        
        # Initialize wandb if available
        if WANDB_AVAILABLE and self.config.logging.use_wandb:
            wandb.init(
                project=self.config.logging.wandb_project,
                entity=self.config.logging.wandb_entity,
                name=self.config.logging.experiment_name,
                config={
                    'model': self.config.model.__dict__,
                    'training': self.config.training.__dict__,
                    'data': self.config.data.__dict__
                }
            )
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log epoch results
            self.logger.info("-" * 60)
            self.logger.info(f"Epoch {epoch + 1}/{self.config.training.num_epochs} Results:")
            self.logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            self.logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            self.logger.info(f"  BLEU-1: {val_metrics.get('bleu_1', 0):.4f}")
            self.logger.info(f"  BLEU-4: {val_metrics.get('bleu_4', 0):.4f}")
            self.logger.info(f"  ROUGE-L: {val_metrics.get('rouge_l', 0):.4f}")
            self.logger.info(f"  METEOR: {val_metrics.get('meteor', 0):.4f}")
            self.logger.info("-" * 60)
            
            # Check if beating baseline
            current_bleu4 = val_metrics.get('bleu_4', 0)
            if current_bleu4 > BASELINE_METRICS['success_threshold']['bleu_4']:
                self.logger.info(f"✓ BLEU-4 ({current_bleu4:.4f}) beats baseline ({BASELINE_METRICS['success_threshold']['bleu_4']})")
            
            # Track training history
            epoch_metrics = {**train_metrics, **val_metrics, 'epoch': epoch + 1}
            self.training_history.append(epoch_metrics)
            
            # Log to wandb
            if WANDB_AVAILABLE and self.config.logging.use_wandb:
                wandb.log(epoch_metrics)
            
            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint('best_loss.pt', val_metrics)
            
            if current_bleu4 > self.best_bleu4:
                self.best_bleu4 = current_bleu4
                self.save_checkpoint('best_bleu4.pt', val_metrics)
            
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt', val_metrics)
            
            # Early stopping
            early_stop_metric = val_metrics.get(
                self.config.training.early_stopping_metric, 
                val_metrics['val_loss']
            )
            if self.early_stopping(early_stop_metric):
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Save final model and training history
        self.save_checkpoint('final_model.pt', val_metrics)
        self.save_training_history()
        
        self.logger.info("=" * 60)
        self.logger.info("Training Complete!")
        self.logger.info(f"Best Val Loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Best BLEU-4: {self.best_bleu4:.4f}")
        self.logger.info("=" * 60)
        
        if WANDB_AVAILABLE and self.config.logging.use_wandb:
            wandb.finish()
        
        return self.training_history
    
    def save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.config.logging.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': {
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__
            }
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch + 1}")
    
    def save_training_history(self):
        """Save training history to JSON file."""
        history_path = os.path.join(
            self.config.logging.log_dir,
            f'{self.config.logging.experiment_name}_history.json'
        )
        
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        self.logger.info(f"Training history saved: {history_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MedTeller Training Script')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration JSON file')
    
    # Override common parameters
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--no_fp16', action='store_true',
                        help='Disable mixed precision training')
    
    # Logging
    parser.add_argument('--wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    """Main entry point for training."""
    args = parse_args()
    
    # Load or create config
    if args.config is not None:
        config = Config.load(args.config)
    else:
        config = get_config()
    
    # Override config with command line arguments
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.data_dir is not None:
        config.data.data_dir = args.data_dir
        config.data.__post_init__()
    if args.output_dir is not None:
        config.logging.output_dir = args.output_dir
        config.logging.__post_init__()
    if args.experiment_name is not None:
        config.logging.experiment_name = args.experiment_name
        config.logging.__post_init__()
    if args.device is not None:
        config.device = args.device
    if args.fp16:
        config.training.fp16 = True
    if args.no_fp16:
        config.training.fp16 = False
    if args.wandb:
        config.logging.use_wandb = True
    if args.seed is not None:
        config.seed = args.seed
    
    # Set seed
    set_seed(config.seed)
    
    # Setup logging
    logger = setup_logging(config.logging.log_dir, config.logging.experiment_name)
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Data directory: {config.data.data_dir}")
    logger.info(f"  Output directory: {config.logging.output_dir}")
    logger.info(f"  Experiment: {config.logging.experiment_name}")
    
    # Save config
    config_save_path = os.path.join(config.logging.log_dir, 'config.json')
    config.save(config_save_path)
    logger.info(f"Configuration saved: {config_save_path}")
    
    # Create dataloaders
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = MedTeller(config)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger
    )
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    
    # Train!
    training_history = trainer.train()
    
    # Final evaluation on test set
    logger.info("Running final evaluation on test set...")
    test_metrics = trainer.evaluator.evaluate_test_set(
        model=model,
        test_loader=test_loader,
        device=trainer.device
    )
    
    logger.info("=" * 60)
    logger.info("Final Test Set Results:")
    logger.info(f"  BLEU-1: {test_metrics.get('bleu_1', 0):.4f}")
    logger.info(f"  BLEU-2: {test_metrics.get('bleu_2', 0):.4f}")
    logger.info(f"  BLEU-4: {test_metrics.get('bleu_4', 0):.4f}")
    logger.info(f"  ROUGE-L: {test_metrics.get('rouge_l', 0):.4f}")
    logger.info(f"  METEOR: {test_metrics.get('meteor', 0):.4f}")
    
    if test_metrics.get('bleu_4', 0) > BASELINE_METRICS['success_threshold']['bleu_4']:
        logger.info(f"✓ SUCCESS! BLEU-4 beats baseline threshold!")
    else:
        logger.info(f"✗ BLEU-4 below baseline threshold")
    logger.info("=" * 60)
    
    # Save final test metrics
    test_metrics_path = os.path.join(config.logging.log_dir, 'test_metrics.json')
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Test metrics saved: {test_metrics_path}")


if __name__ == '__main__':
    main()


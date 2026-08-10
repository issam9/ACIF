import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

import gc

import torch

from dataset import RealLibriSpeechDataModule
from acif import ACIF


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    
    parser.add_argument("--max_tokens_per_batch", type=int, default=8192, help="Max frames per batch for dynamic batching")
    
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # Loss Weights
    parser.add_argument("--weight_mse", type=float, default=0.01)
    parser.add_argument("--weight_cos", type=float, default=10.0)
    parser.add_argument("--weight_qua", type=float, default=1.0)
    
    parser.add_argument("--weight_ce", type=float, default=0.0, help="Weight for direct Cross Entropy loss")
    parser.add_argument("--weight_kd", type=float, default=0.0, help="Weight for Knowledge Distillation loss")
    parser.add_argument("--kd_layers", type=int, default=1, help="Layer index to extract for KD (-1 for full model)")
    parser.add_argument("--kd_loss_type", type=str, default="kl", choices=["kl", "cosine"], help="Loss type for KD (kl or cosine)")
    
    parser.add_argument("--enc_layers", type=int, default=6, help="Number of projector encoder layers.")

    parser.add_argument("--max_steps", type=int, default=250000, help="Total number of training steps")
    
    parser.add_argument("--embeddings_dir", type=str, default=None, help="Path to pre-computed .pt shards")

    # --- FINE-TUNING CHECKPOINT ARGUMENT ---
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="Path to checkpoint for FINE-TUNING (loads weights only, resets optimizer/steps)")

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    datamodule = RealLibriSpeechDataModule(
        target_hours=None, 
        batch_size=args.batch_size if hasattr(args, 'batch_size') else 1,
        num_workers=args.num_workers,
        embeddings_dir=args.embeddings_dir,
        max_tokens_per_batch=args.max_tokens_per_batch
    )
    
    use_precomputed = args.embeddings_dir is not None
    
    model = ACIF(
        model_name=args.model_name, 
        lr=args.lr, 
        use_precomputed=use_precomputed, 
        weight_mse=args.weight_mse, 
        weight_cos=args.weight_cos, 
        weight_qua=args.weight_qua,
        weight_ce=args.weight_ce,
        weight_kd=args.weight_kd,
        kd_layers=args.kd_layers,         
        kd_loss_type=args.kd_loss_type,   
        enc_layers=args.enc_layers,
    )
    
    prefix = ""
    if args.pretrained_ckpt:
        print(f"\n[INFO] Loading pre-trained weights from {args.pretrained_ckpt} for fine-tuning...")
        checkpoint = torch.load(args.pretrained_ckpt, map_location="cpu", weights_only=True)
        # Handle both raw state_dicts and PyTorch Lightning checkpoints
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        prefix = "ft-"

    ckpt_dir = f"checkpoints/{prefix}{os.path.basename(args.embeddings_dir or 'raw')}-{os.path.basename(args.model_name)}-{args.enc_layers}-{args.lr}-{args.max_steps}-ce{args.weight_ce}-kd{args.weight_kd}-l{args.kd_layers}-{args.weight_mse}-{args.weight_cos}-{args.weight_qua}"
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-step{step}-{train_loss:.3f}",
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_train_steps=20000,  
        save_weights_only=False,
    )

    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    total_batches = len(train_loader)
    print("epochs: ", args.max_steps // total_batches)
    
    trainer = pl.Trainer(
        max_steps=args.max_steps,    
        max_epochs=-1,
        accelerator="gpu",
        strategy="auto",
        devices="auto",
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=100,
        logger=False,
        use_distributed_sampler=False, 
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=100)]
    )
    
    trainer.fit(model, datamodule=datamodule)
import os
import re
import json
import html
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

from acif.utils import normalize_transcript

def load_embedding_layer(model_id, device):
    print(f"Loading tokenizer and embedding layer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load model to CPU first to avoid VRAM spikes
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cpu")
    
    # Extract just the embedding layer and move it to GPU
    embedding_layer = model.get_input_embeddings().to(device)
    embedding_layer.eval()
    
    # Delete the rest of the model and force garbage collection
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return tokenizer, embedding_layer

def precompute_embeddings(output_dir, batch_size=32, shard_size=1000):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    llama_id = "meta-llama/Llama-3.1-8B-Instruct" 
    qwen_id = "Qwen/Qwen2.5-7B-Instruct"
    
    # Load tokenizers and exclusively the embedding layers
    llama_tok, llama_emb = load_embedding_layer(llama_id, device)
    qwen_tok, qwen_emb = load_embedding_layer(qwen_id, device)

    print("\nLoading LibriSpeech train.100 dataset...")
    dataset = load_dataset("openslr/librispeech_asr", "clean", split="train.100")

    current_shard = []
    shard_idx = 0
    total_processed = 0
    
    all_llama_lengths = []
    all_qwen_lengths = []

    print("Extracting text embeddings...")
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]
        raw_transcripts = batch["text"]
        
        # Apply normalization
        norm_transcripts = [normalize_transcript(t) for t in raw_transcripts]

        # Tokenize
        llama_inputs = llama_tok(norm_transcripts, return_tensors="pt", padding=True, truncation=False).to(device)
        qwen_inputs = qwen_tok(norm_transcripts, return_tensors="pt", padding=True, truncation=False).to(device)
        
        with torch.no_grad():
            # Get embeddings (Shape: [batch_size, seq_len, hidden_dim])
            llama_outputs = llama_emb(llama_inputs.input_ids)
            qwen_outputs = qwen_emb(qwen_inputs.input_ids)

        for j in range(len(raw_transcripts)):
            # Determine actual unpadded lengths using attention masks
            llama_valid_len = llama_inputs.attention_mask[j].sum().item()
            qwen_valid_len = qwen_inputs.attention_mask[j].sum().item()
            
            # Slice to unpadded lengths and move to CPU to save VRAM/RAM
            unpadded_llama = llama_outputs[j, :llama_valid_len, :].to(dtype=torch.bfloat16, device="cpu")
            unpadded_qwen = qwen_outputs[j, :qwen_valid_len, :].to(dtype=torch.bfloat16, device="cpu")
            
            all_llama_lengths.append(llama_valid_len)
            all_qwen_lengths.append(qwen_valid_len)
            
            current_shard.append({
                "raw_transcript": raw_transcripts[j],
                "normalized_transcript": norm_transcripts[j],
                "llama_embeddings": unpadded_llama,
                "qwen_embeddings": unpadded_qwen
            })
            total_processed += 1

            if len(current_shard) >= shard_size:
                shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.pt")
                torch.save(current_shard, shard_path)
                current_shard = []
                shard_idx += 1
    
    # Save any remaining samples in the final shard
    if len(current_shard) > 0:
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.pt")
        torch.save(current_shard, shard_path)
        shard_idx += 1

    metadata = {
        "dataset_split": "train.100",
        "total_samples": total_processed,
        "num_shards": shard_idx,
        "shard_size": shard_size,
        "llama_hidden_size": llama_emb.weight.shape[1],
        "qwen_hidden_size": qwen_emb.weight.shape[1],
        "llama_lengths": all_llama_lengths,
        "qwen_lengths": all_qwen_lengths
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print(f"\nDone! Pre-computed {total_processed} samples across {shard_idx} shards.")
    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the .pt shards")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for tokenization and embedding")
    args = parser.parse_args()
    
    precompute_embeddings(args.output_dir, batch_size=args.batch_size)
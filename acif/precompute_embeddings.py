import os
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from transformers import AutoProcessor, SeamlessM4Tv2Model

def precompute_embeddings(output_dir, batch_size=8, shard_size=1000):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").speech_encoder

    model = model.to(device=device, dtype=torch.bfloat16).eval()

    print("Loading the FULL 960-hour LibriSpeech dataset (100 + 360 + 500)...")
    ds_100 = load_dataset("openslr/librispeech_asr", "clean", split="train.100")
    ds_360 = load_dataset("openslr/librispeech_asr", "clean", split="train.360")
    ds_500 = load_dataset("openslr/librispeech_asr", "other", split="train.500")
    
    print("Concatenating and shuffling...")
    dataset = concatenate_datasets([ds_100, ds_360, ds_500])
    dataset = dataset.shuffle(seed=123)

    current_shard = []
    shard_idx = 0
    total_processed = 0
    
    all_lengths = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]
        audio_arrays = [item["array"] for item in batch["audio"]]
        transcripts = batch["text"]

        inputs = processor(
            audios=audio_arrays, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True, 
            return_attention_mask=True
        )
            
        inputs = inputs.to(device=device, dtype=torch.bfloat16)
        
        with torch.no_grad():
            encoder_outputs = model(
                input_features=inputs.input_features, 
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )
            sfm_features = encoder_outputs.hidden_states[24]
            ratio = inputs.input_features.size(1) / sfm_features.size(1)
            valid_lengths = torch.round(inputs.attention_mask.sum(dim=1).float() / ratio).long()
            

        for j in range(len(audio_arrays)):
            valid_len = valid_lengths[j].item()
            unpadded_embed = sfm_features[j, :valid_len, :].to(dtype=torch.bfloat16, device="cpu")
            
            all_lengths.append(valid_len)
            
            current_shard.append({
                "sfm_features": unpadded_embed,
                "transcript": transcripts[j]
            })
            total_processed += 1

            if len(current_shard) >= shard_size:
                shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.pt")
                torch.save(current_shard, shard_path)
                current_shard = []
                shard_idx += 1
    
    if len(current_shard) > 0:
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.pt")
        torch.save(current_shard, shard_path)
        shard_idx += 1

    metadata = {
        "dataset_config": "full_960h",
        "total_samples": total_processed,
        "num_shards": shard_idx,
        "shard_size": shard_size,
        "sfm_dim": model.config.hidden_size,
        "lengths": all_lengths  
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print(f"\nDone! Pre-computed {total_processed} samples across {shard_idx} shards.")
    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the .pt shards")
    args = parser.parse_args()
    
    precompute_embeddings(args.output_dir)
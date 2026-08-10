import os
import json
import random
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Sampler
from datasets import load_dataset
from transformers import AutoProcessor


class ShardAwareDynamicBatchSampler(Sampler):
    def __init__(self, dataset_len, shard_size, lengths, max_tokens, shuffle=True):
        self.dataset_len = dataset_len
        self.shard_size = shard_size
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        
        self.num_shards = (dataset_len + shard_size - 1) // shard_size
        self.shard_to_batches, self.total_batches = self._build_batches()

    def _build_batches(self):
        shard_to_batches = {}
        total_batches = 0
        
        for shard_id in range(self.num_shards):
            start = shard_id * self.shard_size
            end = min(start + self.shard_size, self.dataset_len)
            
            # 1. Get valid indices for this shard
            indices = list(range(start, end))
            
            # 2. Sort indices by sequence length to minimize padding
            indices.sort(key=lambda x: self.lengths[x])
            
            # 3. Dynamic Bucketing: Pack until max_tokens is reached
            shard_batches = []
            current_batch = []
            max_len = 0
            
            for idx in indices:
                max_len = max(max_len, self.lengths[idx])
                
                if max_len * (len(current_batch) + 1) > self.max_tokens and len(current_batch) > 0:
                    shard_batches.append(current_batch)
                    current_batch = [idx]
                    max_len = self.lengths[idx]
                else:
                    current_batch.append(idx)
                    
            if current_batch:
                shard_batches.append(current_batch)
                
            shard_to_batches[shard_id] = shard_batches
            total_batches += len(shard_batches)
            
        return shard_to_batches, total_batches

    def __iter__(self):
        shard_ids = list(range(self.num_shards))
        if self.shuffle:
            random.shuffle(shard_ids)
            
        for shard_id in shard_ids:
            shard_batches = self.shard_to_batches[shard_id]
            
            if self.shuffle:
                random.shuffle(shard_batches)
                
            for batch in shard_batches:
                yield batch

    def __len__(self):
        return self.total_batches


class PrecomputedLibriDataset(Dataset):
    def __init__(self, embeddings_dir, target_hours=None):
        self.embeddings_dir = embeddings_dir
        with open(os.path.join(embeddings_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
            
        self.shard_size = self.metadata["shard_size"]
        self.lengths = self.metadata.get("lengths", [])
        if not self.lengths:
            self.lengths = [500] * self.metadata["total_samples"]
        
        total_samples = self.metadata["total_samples"]
        if target_hours is not None:
            max_samples = int((target_hours * 3600) / 12.0)
            self.dataset_length = min(max_samples, total_samples)
        else:
            self.dataset_length = total_samples
            
        self.active_shard_idx = -1
        self.active_shard_data = None

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        shard_idx = idx // self.shard_size
        item_idx = idx % self.shard_size
        
        if shard_idx != self.active_shard_idx:
            shard_path = os.path.join(self.embeddings_dir, f"shard_{shard_idx:04d}.pt")
            self.active_shard_data = torch.load(shard_path, map_location="cpu", weights_only=True)
            self.active_shard_idx = shard_idx
            
        return self.active_shard_data[item_idx]


class RealLibriSpeechDataModule(pl.LightningDataModule):
    def __init__(self, target_hours: float = 10.0, batch_size: int = 2, num_workers: int = 4, embeddings_dir: str = None, max_tokens_per_batch: int = 4000):
        super().__init__()
        self.target_hours = target_hours
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.embeddings_dir = embeddings_dir
        
        self.max_tokens_per_batch = max_tokens_per_batch
        
        if not self.embeddings_dir:
            self.processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")

    def setup(self, stage=None):
        if self.embeddings_dir:
            print(f"Loading PRE-COMPUTED dataset from {self.embeddings_dir}...")
            self.train_dataset = PrecomputedLibriDataset(self.embeddings_dir, self.target_hours)
            print(f"Data ready. Serving {len(self.train_dataset)} pre-computed samples.")
        else:
            print("Loading RAW openslr/librispeech_asr dataset...")
            raw_ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100").shuffle(seed=42)
            
            selected_indices = []
            accumulated_seconds = 0.0
            target_seconds = self.target_hours * 3600.0

            for idx, item in enumerate(raw_ds):
                duration = len(item['audio']['array']) / item['audio']['sampling_rate']
                if duration > 15.0:
                    continue
                accumulated_seconds += duration
                selected_indices.append(idx)
                if accumulated_seconds >= target_seconds:
                    break
                    
            self.train_dataset = raw_ds.select(selected_indices)
            print(f"Data ready. Downsampled pool holds {len(self.train_dataset)} samples.")

    def train_dataloader(self):
        if self.embeddings_dir:
            print(f"Applying Shard-Aware Dynamic Batching (Max Tokens: {self.max_tokens_per_batch})...")
            
            batch_sampler = ShardAwareDynamicBatchSampler(
                dataset_len=len(self.train_dataset),
                shard_size=self.train_dataset.shard_size,
                lengths=self.train_dataset.lengths,
                max_tokens=self.max_tokens_per_batch,
                shuffle=True
            )
            
            return DataLoader(
                self.train_dataset, 
                batch_sampler=batch_sampler, 
                collate_fn=self.collate_fn, 
                num_workers=self.num_workers
            )
        else:
            # Fallback to standard batching for raw audio
            return DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                collate_fn=self.collate_fn, 
                num_workers=self.num_workers, 
                shuffle=True, 
                drop_last=True
            )

    def collate_fn(self, batch):
        if "sfm_features" in batch[0]: 
            sfm_features = [item["sfm_features"] for item in batch]
            transcripts = [item["transcript"] for item in batch]
            sfm_lengths = torch.tensor([f.size(0) for f in sfm_features], dtype=torch.long)
            
            padded_sfm = torch.nn.utils.rnn.pad_sequence(sfm_features, batch_first=True)
            return {"sfm_features": padded_sfm, "sfm_lengths": sfm_lengths, "transcripts": transcripts}
            
        else: 
            audio_arrays = [item['audio']['array'] for item in batch]
            transcripts = [item['text'] for item in batch]
            audio_inputs = self.processor(audio=audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True)
            return {"input_features": audio_inputs.input_features, "attention_mask": audio_inputs.attention_mask, "transcripts": transcripts}
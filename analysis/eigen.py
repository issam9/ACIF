import os
import glob
import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm

def compute_pca_variance_decay(embeddings: torch.Tensor):
    """
    Computes the cumulative variance explained by the principal components 
    of the embedding space.
    """
    # 1. Center the embeddings around the mean 
    centered_embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    
    # 2. Compute the covariance matrix
    # Shape: (d, d)
    cov_matrix = (centered_embeddings.T @ centered_embeddings) / (embeddings.size(0) - 1)
    
    # 3. Compute eigenvalues and eigenvectors
    eigenvalues, _ = torch.linalg.eigh(cov_matrix)
    
    # 4. Sort eigenvalues in descending order
    eigenvalues = torch.sort(eigenvalues, descending=True).values
    
    # 5. Calculate cumulative variance explained
    total_variance = torch.sum(eigenvalues)
    explained_variance = eigenvalues / total_variance
    cumulative_variance = torch.cumsum(explained_variance, dim=0)
    
    return (
        cumulative_variance.cpu().numpy(),
        eigenvalues.cpu().numpy(),
    )


def analyze_modality_gap(llama_embs: torch.Tensor, qwen_embs: torch.Tensor):
    """
    Plots the PCA variance decay.
    """
    
    save_name = "embedding_pca.pdf"
    
    decay_llama, eig_llama = compute_pca_variance_decay(llama_embs)
    decay_qwen, eig_qwen = compute_pca_variance_decay(qwen_embs)
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (3.5, 2.5), 
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
    })
    
    fig, ax = plt.subplots()
    
    color_llama = '#0072B2' 
    color_qwen = '#D55E00'
    
    ax.plot(decay_llama, label='Llama 3.1', linestyle='-', color=color_llama) 
    ax.plot(decay_qwen, label='Qwen 2.5', linestyle='--', color=color_qwen) 
    
    ax.set_xlabel("Principal Components")
    ax.set_ylabel("Cumulative Variance")
    
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    max_dim = max(llama_embs.shape[1], qwen_embs.shape[1])
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max_dim)
    
    formatter = ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x >= 1000 else str(int(x)))
    ax.xaxis.set_major_formatter(formatter)
    
    ax.legend(loc='lower right', frameon=False)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved successfully as '{save_name}'.")
    plt.show()
    

def load_embeddings_from_dir(input_dir):
    """
    Loads all .pt shards from the input directory, extracts the embeddings,
    and flattens them into a continuous point cloud of all tokens.
    """
    shard_files = sorted(glob.glob(os.path.join(input_dir, "shard_*.pt")))
    if not shard_files:
        raise ValueError(f"No shard_*.pt files found in {input_dir}")
    
    all_llama = []
    all_qwen = []
    
    print(f"Loading {len(shard_files)} shards from {input_dir}...")
    for shard_file in tqdm(shard_files):
        shard_data = torch.load(shard_file, map_location="cpu", weights_only=False)
        for item in shard_data:
            all_llama.append(item["llama_embeddings"].to(torch.float32))
            all_qwen.append(item["qwen_embeddings"].to(torch.float32))
            
    llama_tensor = torch.cat(all_llama, dim=0)
    qwen_tensor = torch.cat(all_qwen, dim=0)
    
    print(f"Total Llama tokens loaded: {llama_tensor.shape[0]:,}, Dimension: {llama_tensor.shape[1]}")
    print(f"Total Qwen tokens loaded:  {qwen_tensor.shape[0]:,}, Dimension: {qwen_tensor.shape[1]}")
    
    return llama_tensor, qwen_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the saved .pt shards")
    args = parser.parse_args()
    
    # 1. Load the actual embeddings from disk
    llama_real, qwen_real = load_embeddings_from_dir(args.input_dir)
    
    # 2. Run the analysis!
    analyze_modality_gap(llama_real, qwen_real)
import torch
import torch.nn.functional as F

import re
import html
import unicodedata

def lens_to_padding_m(lens, max_lens):
    bsz = lens.size(0)
    mask = torch.arange(max_lens, device=lens.device).unsqueeze(0).expand(bsz, -1)
    mask = mask >= lens.view(bsz, 1)
    return mask

def distance(x, y, type="cosine"):
    if type == "L2":
        return torch.cdist(x, y)
    else:
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)
        return torch.bmm(x_norm, y_norm.transpose(-1, -2))

def compute_trellis(x, y, x_padding_mask, y_padding_mask):
    sim = distance(x, y, type="cosine")
    sim = sim.masked_fill(x_padding_mask.unsqueeze(-1), -float("inf"))
    sim = sim.masked_fill(y_padding_mask.unsqueeze(-2), -float("inf"))
    num_frame = sim.shape[1] 

    trellis = torch.zeros_like(sim)
    trellis[:, :, 0] = torch.cumsum(sim[:, :, 0], 1)
    
    trellis[:, 0, 1:] = -float("inf")   
    
    valid_x = (~x_padding_mask).sum(dim=1)
    valid_y = (~y_padding_mask).sum(dim=1)
    lengths = valid_x - valid_y
    
    mask = lens_to_padding_m(lengths + 1, x_padding_mask.shape[-1])
    trellis[:, :, 0] = trellis[:, :, 0].masked_fill(mask, -float("inf"))

    for t in range(num_frame - 1):
        trellis[:, t + 1, 1:] = torch.maximum(
            trellis[:, t, 1:],      
            trellis[:, t, :-1]      
        ) + sim[:, t+1, 1:]
        
    return trellis

def backtrack(trellis, x_padding_mask, y_padding_mask):
    B, T_x, T_y = trellis.shape
    device = trellis.device

    num_frames = (~x_padding_mask).sum(dim=1) - 1
    num_tokens = (~y_padding_mask).sum(dim=1) - 1

    align = torch.full((B, T_x), -1, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device)

    initial_t = num_frames.clone().clamp(min=0)
    initial_j = num_tokens.clone().clamp(min=0) 
    valid_end_mask = num_frames >= 0

    if valid_end_mask.any():
        align[batch_indices[valid_end_mask], initial_t[valid_end_mask]] = initial_j[valid_end_mask]

    current_j = initial_j.clone()
    for t_decision in range(T_x - 1, -1, -1):
        active_mask = (initial_t > t_decision)

        stayed_indices = current_j.clamp(min=0, max=T_y-1)
        changed_indices = (current_j - 1).clamp(min=0, max=T_y-1)
        
        stayed_scores = trellis[batch_indices, t_decision, stayed_indices]
        changed_scores = trellis[batch_indices, t_decision, changed_indices]

        change_impossible_mask = (current_j == 0) | (~active_mask)
        changed_scores = changed_scores.masked_fill(change_impossible_mask, -float("inf"))
        stayed_scores = stayed_scores.masked_fill(~active_mask, -float("inf")) 

        decision_mask = changed_scores > stayed_scores
        next_j = torch.where(decision_mask & active_mask, current_j - 1, current_j)

        align[batch_indices, t_decision] = next_j 
        current_j = next_j

    align[:, 0] = 0
    align.masked_fill_(x_padding_mask, -1)

    return align


def normalize_transcript(text: str) -> str:
    # 1. HTML Clean-up
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. Unicode NFKC Normalization
    text = unicodedata.normalize('NFKC', text)
    
    # 3. Lowercasing
    text = text.lower()
    
    # 4. Punctuation Removal
    text = re.sub(r'[^\w\s]', '', text)
    
    # 5. Collapse multiple spaces into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, SeamlessM4Tv2Model
from transformers import get_cosine_schedule_with_warmup

from modules import ContinuousIntegrateAndFire, ACIFEncoder
from utils import compute_trellis, backtrack, normalize_transcript


class BaseAudioLLMSystem(pl.LightningModule):
    def __init__(self, model_name: str, lr: float = 1e-4, use_precomputed: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.use_precomputed = use_precomputed
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if not self.use_precomputed:
            seamless_base = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
            self.speech_encoder = seamless_base.speech_encoder
            self.speech_encoder.eval()
            for param in self.speech_encoder.parameters():
                param.requires_grad = False
            self.sfm_dim = self.speech_encoder.config.hidden_size
        else:
            self.sfm_dim = 1024  

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "speech_encoder"):
            self.speech_encoder.eval()
        if hasattr(self, "llm"):
            self.llm.eval()
        return self

    def on_fit_start(self):
        if hasattr(self, "speech_encoder"):
            self.speech_encoder = self.speech_encoder.to(self.device)
        if hasattr(self, "embed_layer"):
            self.embed_layer = self.embed_layer.to(self.device)
        if hasattr(self, "llm"):
            self.llm = self.llm.to(self.device)

    def on_save_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        trainable_keys = {name for name, param in self.named_parameters() if param.requires_grad}
        buffer_keys = {name for name, buf in self.named_buffers()}
        keys_to_keep = trainable_keys | buffer_keys
        checkpoint['state_dict'] = {k: v for k, v in state_dict.items() if k in keys_to_keep}

    def get_speech_features_and_lengths(self, batch):
        if self.use_precomputed:
            # Cast dynamically from saved bfloat16 to whatever the trainer is currently using
            sfm_features = batch["sfm_features"].to(device=self.device, dtype=self.dtype)
            sfm_lengths = batch["sfm_lengths"].to(self.device)
            return sfm_features, sfm_lengths
        else:
            encoder_outputs = self.speech_encoder(
                input_features=batch["input_features"].to(self.device), 
                attention_mask=batch["attention_mask"].to(self.device)
            )
            sfm_features = encoder_outputs.last_hidden_state
            
            ratio = batch["input_features"].size(1) / sfm_features.size(1)
            sfm_lengths = torch.round(batch["attention_mask"].sum(dim=1).float() / ratio).long()
            return sfm_features, sfm_lengths
        

class ACIF(BaseAudioLLMSystem):
    def __init__(self, model_name: str, lr: float = 1e-5, use_precomputed: bool = False, weight_mse: float = 5.0, 
                 weight_cos: float = 10.0, weight_qua: float = 1.0, weight_ce: float = 0.0,
                 weight_kd: float = 0.0, kd_layers: int = 1, kd_loss_type: str = "kl", enc_layers: int = 6):
        super().__init__(model_name, lr, use_precomputed)
        self.weight_mse = weight_mse
        self.weight_cos = weight_cos
        self.weight_qua = weight_qua
        self.weight_ce = weight_ce
        
        # Knowledge Distillation Params
        self.weight_kd = weight_kd
        self.kd_layers = kd_layers
        self.kd_loss_type = kd_loss_type
        
        temp_llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=None)
        raw_embed_weight = temp_llm.get_input_embeddings().weight.clone().detach()
        vocab_size, self.llm_dim = raw_embed_weight.shape
        
        self.embed_layer = nn.Embedding(vocab_size, self.llm_dim)
        self.embed_layer.weight = nn.Parameter(raw_embed_weight)
        self.embed_layer.requires_grad_(False)
        
        # CONDITIONAL LLM RETENTION & PRUNING
        if self.weight_ce > 0.0 or self.weight_kd > 0.0:
            self.llm = temp_llm
            if self.weight_ce == 0.0 and self.weight_kd > 0.0:
                if hasattr(self.llm.model, 'layers'):
                    self.llm.model.layers = self.llm.model.layers[:self.kd_layers]
                    print(f"Pruned LLM down to {self.kd_layers} layers for efficient Knowledge Distillation.")
            self.llm.requires_grad_(False)
            self.llm.eval()
        else:
            del temp_llm
            gc.collect()
        
        self.encoder = ACIFEncoder(
            in_channels=self.sfm_dim,
            transformer_dim=1024,
            num_layers=enc_layers
        )
        
        self.cif = ContinuousIntegrateAndFire(in_dim=1024, out_dim=self.llm_dim)

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            sfm_features, sfm_lengths = self.get_speech_features_and_lengths(batch)
            clean_transcripts = [normalize_transcript(t) for t in batch["transcripts"]]
            tokens = self.tokenizer(clean_transcripts, return_tensors="pt", padding=True, add_special_tokens=False)
            gt_embeds = self.embed_layer(tokens.input_ids.to(self.device))
            
        features, sfm_lengths = self.encoder(sfm_features, sfm_lengths)
        
        frame_proj, alphas = self.cif(features)
        frame_proj = frame_proj.to(gt_embeds.dtype)
        
        B, T_x, D = frame_proj.shape
        _, T_y, _ = gt_embeds.shape
        
        text_lengths = tokens.attention_mask.sum(dim=1).to(self.device)
        x_padding_mask = torch.arange(T_x, device=self.device).unsqueeze(0).expand(B, T_x) >= sfm_lengths.unsqueeze(1)
        y_padding_mask = torch.arange(T_y, device=self.device).unsqueeze(0).expand(B, T_y) >= text_lengths.unsqueeze(1)
        
        with torch.no_grad():
            norm_frame_proj = F.normalize(frame_proj.float(), p=2, dim=-1, eps=1e-8)
            norm_gt_embeds = F.normalize(gt_embeds.float(), p=2, dim=-1, eps=1e-8)
            
            trellis = compute_trellis(norm_frame_proj, norm_gt_embeds, x_padding_mask, y_padding_mask)
            align = backtrack(trellis, x_padding_mask, y_padding_mask)
        
        valid_mask = (align != -1)
        safe_align = align.clamp(min=0)
        
        # DTW SEGMENT-WISE POOLING (Alpha-Gated Weighted Average)
        sum_embeds = torch.zeros((B, T_y, D), device=self.device, dtype=frame_proj.dtype)
        alpha_sums = torch.zeros((B, T_y, 1), device=self.device, dtype=frame_proj.dtype)
        
        # We keep counts strictly for the boolean valid_tokens_mask
        counts = torch.zeros((B, T_y, 1), device=self.device, dtype=torch.long)
        
        align_exp = safe_align.unsqueeze(-1).expand(B, T_x, D)
        valid_mask_exp = valid_mask.unsqueeze(-1).expand(B, T_x, D)
        
        # Multiply by alphas
        weighted_frame_proj = frame_proj * alphas.unsqueeze(-1)
        
        # Accumulate the weighted features
        sum_embeds.scatter_add_(1, align_exp, weighted_frame_proj * valid_mask_exp.to(frame_proj.dtype))
        
        # Accumulate the total alpha weight per token for perfect scaling
        alpha_sums.scatter_add_(1, safe_align.unsqueeze(-1), alphas.unsqueeze(-1) * valid_mask.unsqueeze(-1).to(alphas.dtype))
        
        # Accumulate raw frame counts purely for the existence mask
        counts.scatter_add_(1, safe_align.unsqueeze(-1), valid_mask.unsqueeze(-1).to(counts.dtype))
        
        # Divide by the alpha sum to get a mathematically perfect weighted mean!
        pooled_embeds = sum_embeds / alpha_sums.clamp(min=1e-5)
        
        # The restored counts tensor now safely masks out empty DTW bins!
        valid_tokens_mask = (~y_padding_mask) & (counts.squeeze(-1) > 0)
        
        matched_pooled = pooled_embeds[valid_tokens_mask]
        matched_gt = gt_embeds[valid_tokens_mask]
        
        # Normalized MSE
        raw_mse = F.mse_loss(matched_pooled, matched_gt)
        target_mag = (matched_gt ** 2).mean()
        l_mse = raw_mse / target_mag.detach().clamp(min=1e-8)
        
        # Mean-Centered Cosine Loss
        target_mean = matched_gt.mean(dim=0, keepdim=True).detach()
        
        # Subtract the center to shift everything to the origin
        centered_pooled = matched_pooled - target_mean
        centered_gt = matched_gt - target_mean
        
        # Calculate cosine similarity on the centered vectors
        cos_sim = F.cosine_similarity(centered_pooled, centered_gt, dim=-1)
        l_cos = 1.0 - cos_sim.mean()
        
        valid_text_mask = ~y_padding_mask
        l_qua = F.l1_loss(alpha_sums[valid_text_mask], torch.ones_like(alpha_sums[valid_text_mask]))

        l_ce = torch.tensor(0.0, device=self.device)
        l_kd = torch.tensor(0.0, device=self.device)
        
        if self.weight_ce > 0.0 or self.weight_kd > 0.0:
            self.llm.eval()
            llm_attention_mask = (~y_padding_mask).long()
            
            # Forward pass speech embeddings
            get_hiddens = (self.weight_kd > 0.0)
            speech_outputs = self.llm(inputs_embeds=pooled_embeds, attention_mask=llm_attention_mask, output_hidden_states=get_hiddens)
            
            # Cross Entropy Loss
            if self.weight_ce > 0.0:
                logits = speech_outputs.logits
                labels = tokens.input_ids.to(self.device).clone()
                labels[y_padding_mask] = -100
                
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                l_ce = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1), 
                    ignore_index=-100
                )
                
            # Knowledge Distillation Loss (Logit Lens or Cosine)
            if self.weight_kd > 0.0:
                with torch.no_grad(): # Get teacher (text) targets
                    text_outputs = self.llm(inputs_embeds=gt_embeds, attention_mask=llm_attention_mask, output_hidden_states=True)
                
                # Safely extract the requested layer (if pruned, -1 gets the final pruned layer)
                layer_idx = min(self.kd_layers, len(speech_outputs.hidden_states) - 1)
                
                s_hiddens = speech_outputs.hidden_states[layer_idx]
                t_hiddens = text_outputs.hidden_states[layer_idx]
                valid_mask_kd = ~y_padding_mask
                
                if self.kd_loss_type == "kl":
                    # LOGIT LENS: Project intermediate hidden states to vocab space
                    norm_fn = getattr(self.llm.model, "norm", getattr(self.llm.model, "layer_norm", None))
                    s_normed = norm_fn(s_hiddens) if norm_fn else s_hiddens
                    t_normed = norm_fn(t_hiddens) if norm_fn else t_hiddens
                    
                    s_logits = self.llm.lm_head(s_normed)[valid_mask_kd]
                    t_logits = self.llm.lm_head(t_normed)[valid_mask_kd]
                    
                    # KL Divergence: student logs vs teacher probabilities
                    l_kd = F.kl_div(
                        F.log_softmax(s_logits, dim=-1),
                        F.softmax(t_logits, dim=-1),
                        reduction='batchmean'
                    )
                elif self.kd_loss_type == "cosine":
                    # DIRECT SIMILARITY: Compare intermediate contextual embeddings
                    s_h_v = s_hiddens[valid_mask_kd]
                    t_h_v = t_hiddens[valid_mask_kd]
                    l_kd = 1.0 - F.cosine_similarity(s_h_v, t_h_v, dim=-1).mean()
        
        total_loss = (self.weight_mse * l_mse) + (self.weight_cos * l_cos) + \
                     (self.weight_qua * l_qua) + (self.weight_ce * l_ce) + (self.weight_kd * l_kd)
             
        self.log_dict({
            "train_loss": total_loss, 
            "l_mse": l_mse, 
            "l_cos": l_cos, 
            "l_qua": l_qua,
            "l_ce": l_ce,
            "l_kd": l_kd,
        }, prog_bar=True, batch_size=B, sync_dist=True, on_step=True, on_epoch=True)
        
        return total_loss

    def configure_optimizers(self):
        trainable_params = list(self.encoder.parameters()) + list(self.cif.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=self.hparams.lr)

        total_steps = self.trainer.max_steps
        warmup_steps = int(self.trainer.max_steps * 0.1)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
 
import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, SeamlessM4Tv2Model
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import torch.nn.functional as F
import gc

from acif.modules import ACIFEncoder, ContinuousIntegrateAndFire

from acif.utils import normalize_transcript

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, choices=["librispeech", "fleurs", "voxpopuli"], default="librispeech", help="Dataset to evaluate on")
    
    parser.add_argument("--method", type=str, required=True, choices=["acif", "baseline"])
    parser.add_argument("--ckpt_path", type=str, default=None, required=False)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate (e.g., test_clean, test, validation)")
    
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers of projector")
    
    args = parser.parse_args()

    if args.dataset == "librispeech":
        if args.split == "dev": args.split = "dev_clean"
        if args.split == "test": args.split = "test_clean"

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INIT] Running zero-shot evaluation on: {device} | Dataset: {args.dataset.upper()} | Method: {args.method} | Split: {args.split.upper()} | Batch Size: {args.batch_size}")

    if args.method != "baseline":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        llm = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
        llm.eval()
        llm_dim = llm.config.hidden_size

    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    seamless_base = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
    
    if args.method == "baseline":
        seamless_base = seamless_base.to(device=device, dtype=torch.bfloat16)
        seamless_base.eval()
    else:
        speech_encoder = seamless_base.speech_encoder.to(device=device, dtype=torch.bfloat16)
        speech_encoder.eval()
        sfm_dim = speech_encoder.config.hidden_size
        del seamless_base
        gc.collect()
        torch.cuda.empty_cache()


    if args.method != "baseline":
        print(f"[INIT] Reconstructing the {args.method} bridge module...")
        if args.ckpt_path is None:
            raise ValueError(f"--ckpt_path is required for method {args.method}")
            
        checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        bridge_modules = {}

        encoder = ACIFEncoder(in_channels=sfm_dim, transformer_dim=1024, num_layers=int(args.num_layers)).to(device)
        cif = ContinuousIntegrateAndFire(in_dim=1024, out_dim=llm_dim).to(device)
        
        enc_sd = {k.replace("encoder.", ""): v for k, v in state_dict.items() if "encoder." in k}
        cif_sd = {k.replace("cif.", ""): v for k, v in state_dict.items() if k.startswith("cif.")}
        
        encoder.load_state_dict(enc_sd, strict=True)
        cif.load_state_dict(cif_sd, strict=True)
        
        bridge_modules["encoder"] = encoder.to(dtype=torch.bfloat16).eval()
        bridge_modules["cif"] = cif.to(dtype=torch.bfloat16).eval()


    if args.dataset == "librispeech":
        ls_config = "clean" if "clean" in args.split else "other"
        split_target = "validation" if "dev" in args.split else "test"
        test_ds = load_dataset("openslr/librispeech_asr", ls_config, split=split_target)
        text_column = "text"
    elif args.dataset == "fleurs":
        split_target = "validation" if "dev" in args.split else "test"
        test_ds = load_dataset("google/fleurs", "en_us", split=split_target)
        text_column = "transcription"
    elif args.dataset == "voxpopuli":
        split_target = "validation" if "dev" in args.split or "validation" in args.split else "test"
        test_ds = load_dataset("facebook/voxpopuli", "en", split=split_target)
        text_column = "normalized_text"
    
    if args.num_samples is not None:
        test_ds = test_ds.select(range(args.num_samples))

    if args.method != "baseline":
        system_prompt = "You are a helpful ASR transcription assistant."
        instruction = "Repeat the previous text between the quotes in its entirety just once and nothing else. Do not repeat the text multiple times or correct the text or add punctuation. End the text if you notice a phrase or a text is getting repeated. Ignore the words that do not make any sense."
        
        model_lower = args.model_name.lower()
        
        if "qwen" in model_lower:
            prefix_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n\""
            postfix_text = f"\"\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        elif "llama" in model_lower:
            prefix_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\""
            postfix_text = f"\"\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            prefix_text = f"System: {system_prompt}\n\nUser: \""
            postfix_text = f"\"\n{instruction}\n\nAssistant: "
        
        prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        postfix_ids = tokenizer(postfix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        
        with torch.no_grad():
            prefix_embeds = llm.get_input_embeddings()(prefix_ids)[0]
            postfix_embeds = llm.get_input_embeddings()(postfix_ids)[0]

            vocab_matrix = llm.get_input_embeddings().weight 
            vocab_norm = F.normalize(vocab_matrix.float(), p=2, dim=-1)

    predictions = []
    references = []
    
    for i in tqdm(range(0, len(test_ds), args.batch_size)):
        batch = test_ds[i : i + args.batch_size]
        audio_arrays = [audio['array'] for audio in batch['audio']]
        
        reference_texts = [str(text).lower() if text else "" for text in batch[text_column]]
        
        B = len(audio_arrays)
        
        with torch.no_grad():
            if args.method == "baseline":
                audio_inputs = processor(audios=audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
                audio_inputs = audio_inputs.to(device=device, dtype=torch.bfloat16)
                
                output_tokens = seamless_base.generate(**audio_inputs, tgt_lang="eng", generate_speech=False)
                batch_preds = processor.batch_decode(output_tokens[0], skip_special_tokens=True)
                pred_texts = [text.strip().lower() for text in batch_preds]
                
                for b in range(B):
                    clean_ref = normalize_transcript(reference_texts[b])
                    clean_pred = normalize_transcript(pred_texts[b])
                    
                    references.append(clean_ref)
                    predictions.append(clean_pred)
            else:
                audio_inputs = processor(audios=audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
                audio_inputs = audio_inputs.to(device=device, dtype=torch.bfloat16)
                
                encoder_outputs = speech_encoder(
                    input_features=audio_inputs.input_features,
                    attention_mask=audio_inputs.attention_mask,
                    output_hidden_states=True
                )
                sfm_features = encoder_outputs.hidden_states[24].to(dtype=torch.bfloat16)
                
                input_lens = audio_inputs.attention_mask.sum(dim=1)
                scale_factor = sfm_features.size(1) / audio_inputs.attention_mask.size(1)
                sfm_lengths = (input_lens * scale_factor).round().long()
                
                features, out_lengths = bridge_modules["encoder"](sfm_features, sfm_lengths)
                frame_proj, alphas = bridge_modules["cif"](features)
                
                speech_embeds_list = []
                for b in range(B):
                    valid_len = out_lengths[b]
                    fp_b = frame_proj[b:b+1, :valid_len, :]
                    alpha_b = alphas[b:b+1, :valid_len]
                    
                    embeds = bridge_modules["cif"].infer_integrate(fp_b, alpha_b, threshold=1.0)[0]
                    speech_embeds_list.append(embeds)
                
                batch_inputs_embeds = []
                max_len = 0
                
                for b in range(B):
                    speech_embeds_b = speech_embeds_list[b].to(dtype=torch.bfloat16)
                    seq = torch.cat([prefix_embeds, speech_embeds_b, postfix_embeds], dim=0)
                    batch_inputs_embeds.append(seq)
                    max_len = max(max_len, seq.size(0))
                    
                inputs_embeds = torch.zeros((B, max_len, llm_dim), dtype=torch.bfloat16, device=device)
                attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
                
                for b in range(B):
                    seq_len = batch_inputs_embeds[b].size(0)
                    inputs_embeds[b, max_len - seq_len:] = batch_inputs_embeds[b]
                    attention_mask[b, max_len - seq_len:] = 1
                    
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)
                
                output_ids = llm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=150,
                    num_beams=args.num_beams,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                pred_texts = [tokenizer.decode(out, skip_special_tokens=True).strip() for out in output_ids]
                
                direct_decoded_texts = []
                for b in range(B):
                    speech_norm = F.normalize(speech_embeds_list[b].float(), p=2, dim=-1)
                    sim = torch.matmul(speech_norm, vocab_norm.T)
                    best_tokens = sim.argmax(dim=-1)
                    
                    clean_ref = normalize_transcript(reference_texts[b])
                    clean_pred = normalize_transcript(pred_texts[b])
                    clean_direct = normalize_transcript(tokenizer.decode(best_tokens))
                    
                    references.append(clean_ref)
                    predictions.append(clean_pred)
                    direct_decoded_texts.append(clean_direct)

    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)

    print("\n" + "="*50)
    print(f"EVALUATION COMPLETE ({len(predictions)} Samples)")
    print(f"Method: {args.method} | Dataset: {args.dataset.upper()} | Split: {args.split.upper()} | LLM: {args.model_name if args.method != 'baseline' else 'None'}")
    print(f"Zero-Shot ASR Word Error Rate (WER)      : {wer * 100:.2f}%")
    print(f"Zero-Shot ASR Character Error Rate (CER) : {cer * 100:.2f}%")
    print("="*50)

    if args.method == "baseline" or args.ckpt_path is None:
        ckpt_dir = os.path.join("checkpoints", "baselines", "seamless_e2e")
    else:
        ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt_path))
        
    if args.dataset == "librispeech":
        dataset_label = "librispeech"
    elif args.dataset == "fleurs":
        dataset_label = "fleurs_en_us"
    elif args.dataset == "voxpopuli":
        dataset_label = "voxpopuli_en"

    res_filename = f"asr_{dataset_label}_{args.split}_{args.method}.json"
    res_path = os.path.join(ckpt_dir, res_filename)

    results_dict = {
        "evaluation_config": {
            "dataset": args.dataset,
            "split": args.split,
            "model_name": args.model_name if args.method != "baseline" else "None",
            "checkpoint_path": args.ckpt_path if args.ckpt_path else "None",
            "method": args.method,
            "num_samples": len(predictions)
        },
        "metrics": {
            "wer": wer * 100,
            "cer": cer * 100 
        }
    }

    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)
        
    print(f"\nSaved evaluation metrics to: {res_path}")
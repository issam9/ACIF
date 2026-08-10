import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, SeamlessM4Tv2Model
from datasets import load_dataset, Dataset
from tqdm import tqdm
import csv

import librosa
import warnings
import zipfile
import io
import gc
import tempfile

from langdetect import detect, LangDetectException
from modules import ACIFEncoder, ContinuousIntegrateAndFire

from sacrebleu.metrics import BLEU
from comet import download_model, load_from_checkpoint
import soundfile as sf


# Prompts
ST_PROMPTS = {
    "de": "Können Sie den Inhalt der Rede in den deutschen Text übersetzen?",
    "es": "¿Puedes traducir el contenido del discurso al texto en español?",
    "fr": "Pouvez-vous traduire le contenu du discours en texte français ?",
    "it": "Puoi tradurre il contenuto del discorso in testo italiano?",
    "zh": "你能把演讲内容翻译成中文吗？"
}

# Mapping for SeamlessM4T language codes
SEAMLESS_LANG_MAP = {
    "de": "deu",
    "es": "spa",
    "fr": "fra",
    "it": "ita",
    "zh": "cmn",
    "en": "eng"
}

ZIP_HANDLES = {}

def get_wav_internal_path(internal_path):
    base_path = os.path.splitext(internal_path)[0]
    
    if base_path.startswith("clips/"):
        base_path = "clips_wav/" + base_path[6:]
    if base_path.startswith("audios/"):
        base_path = "audios_wav/" + base_path[7:]
    
    return base_path + ".wav"

def load_audio_from_zip(zip_path, internal_path, start_time=None, end_time=None, sr=16000):
    if zip_path not in ZIP_HANDLES:
        ZIP_HANDLES[zip_path] = zipfile.ZipFile(zip_path, 'r')
    
    z = ZIP_HANDLES[zip_path]
    wav_path = get_wav_internal_path(internal_path)
    
    with z.open(wav_path) as f:
        audio_bytes = f.read()
        
    with io.BytesIO(audio_bytes) as buf:
        try:
            arr, orig_sr = sf.read(buf, dtype='float32')
            if len(arr.shape) > 1:
                arr = arr.mean(axis=1)
            if orig_sr != sr:
                arr = librosa.resample(arr, orig_sr=orig_sr, target_sr=sr)
        except Exception:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file.flush()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    arr, _ = librosa.load(tmp_file.name, sr=sr, mono=True)
            
        if start_time is not None:
            arr = arr[int(float(start_time) * sr):]
        if end_time is not None and start_time is not None:
            arr = arr[:int((float(end_time) - float(start_time)) * sr)]
            
        return arr

def get_audio_frames_from_zip(zip_path, internal_path, start_time=None, end_time=None, sr=16000):
    if start_time is not None and end_time is not None:
        return int((float(end_time) - float(start_time)) * sr)
    
    if zip_path not in ZIP_HANDLES:
        ZIP_HANDLES[zip_path] = zipfile.ZipFile(zip_path, 'r')
        
    z = ZIP_HANDLES[zip_path]
    wav_path = get_wav_internal_path(internal_path)
    
    with z.open(wav_path) as f:
        audio_bytes = f.read()
        
    with io.BytesIO(audio_bytes) as buf:
        try:
            return int(sf.info(buf).duration * sr)
        except Exception:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file.flush()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return int(librosa.get_duration(path=tmp_file.name) * sr)

def generate_text_translation(sources, args, llm, tokenizer, device):
    tokenizer.padding_side = "left"
    
    prompts = []
    for text in sources:
        system_prompt = "You are a helpful translation assistant."
        instruction = ST_PROMPTS[args.tgt_lang]
        suffix = "Output only the translation and nothing else." 
        
        model_lower = args.model_name.lower()
        if "qwen" in model_lower:
            p = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{instruction}\n\n\"{text}\"\n\n{suffix}<|im_end|>\n<|im_start|>assistant\n"
        elif "llama" in model_lower:
            p = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\n\"{text}\"\n\n{suffix}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            p = f"System: {system_prompt}\n\nUser: {instruction}\n\n\"{text}\"\n\n{suffix}\n\nAssistant: "
        prompts.append(p)
        
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    
    with torch.no_grad():
        output_ids = llm.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=args.num_beams,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    input_length = inputs.input_ids.shape[1]
    new_tokens = output_ids[:, input_length:]
    pred_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return pred_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, choices=["none", "text_llm", "cascaded", "seamless_e2e"], default="none", help="Which baseline to run")
    
    parser.add_argument("--ckpt_path", type=str, default="")
    
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    
    parser.add_argument("--dataset", type=str, choices=["covost2", "europarl_st"], default="europarl_st", help="Dataset to evaluate on")
    parser.add_argument("--tgt_lang", type=str, choices=["de", "es", "fr", "it", "zh"], default="de", help="Target translation language")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate (e.g., dev, test)")
    
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams")
    
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers in the projector encoder")
    
    
    args = parser.parse_args()
    
    # Validation checks
    if args.baseline == "none" and not args.ckpt_path:
        raise ValueError("--ckpt_path is required unless running a baseline.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_device = f"cuda:{torch.cuda.device_count() - 1}" if torch.cuda.device_count() > 1 else device
    
    print(f"\n[INIT] Baseline mode: {args.baseline.upper()}")
    print(f"[INIT] Dataset: {args.dataset} (en->{args.tgt_lang})")
    
    llm, tokenizer, processor, seamless_model, speech_encoder = None, None, None, None, None
    bridge_modules = {}

    # Load models
    if args.baseline in ["none", "text_llm", "cascaded"]:
        print(f"[INIT] Loading LLM on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        llm = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
        llm.eval()
        llm_dim = llm.config.hidden_size

    # Load Audio Encoders (if needed)
    if args.baseline in ["none", "cascaded", "seamless_e2e"]:
        print(f"[INIT] Loading Audio Encoder on {audio_device}...")
        processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        seamless_base = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large", torch_dtype=torch.bfloat16)
        
        if args.baseline == "none":
            speech_encoder = seamless_base.speech_encoder.to(device=audio_device)
            del seamless_base
            gc.collect()
            torch.cuda.empty_cache()
        else:
            # Need the full model for cascaded/e2e generation
            seamless_model = seamless_base.to(device=audio_device)
            seamless_model.eval()
            
    # Load Bridge Module
    if args.baseline == "none":
        sfm_dim = speech_encoder.config.hidden_size
        checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        encoder = ACIFEncoder(in_channels=sfm_dim, transformer_dim=1024, num_layers=int(args.num_layers)).to(audio_device)
        cif = ContinuousIntegrateAndFire(in_dim=1024, out_dim=llm_dim).to(audio_device)
        
        enc_sd = {k.replace("encoder.", ""): v for k, v in state_dict.items() if "encoder." in k}
        cif_sd = {k.replace("cif.", ""): v for k, v in state_dict.items() if k.startswith("cif.")}
        
        encoder.load_state_dict(enc_sd, strict=True)
        cif.load_state_dict(cif_sd, strict=True)
        
        bridge_modules["encoder"] = encoder.to(dtype=torch.bfloat16).eval()
        bridge_modules["cif"] = cif.to(dtype=torch.bfloat16).eval()

    print(f"[INIT] Loading {args.dataset} dataset from local ZIP archives...")
    if args.dataset == "covost2":
        folder_name = f"covost_v2-en_{args.tgt_lang}"
        tsv_filename = f"covost_v2.en_{args.tgt_lang}.{args.split}.tsv"
        tsv_path = os.path.join("data", folder_name, tsv_filename)
        zip_path = os.path.join("data", folder_name, "clips_wav.zip")
        
        zip_paths, internal_paths, sentences, translations = [], [], [], []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                zip_paths.append(zip_path)
                internal_paths.append(os.path.join("clips_wav", row['path']))
                sentences.append(row['sentence'])
                translations.append(row['translation'])
                
        test_ds = Dataset.from_dict({"zip_path": zip_paths, "internal_path": internal_paths, "sentence": sentences, "translation": translations})
    else: 
        full_ds = load_dataset("tj-solergibert/Europarl-ST", split=args.split)
        test_ds = full_ds.filter(
            lambda orig_lang, trans: orig_lang == "en" and trans is not None and args.tgt_lang in trans,
            input_columns=["original_language", "transcriptions"]
        )
        zip_path = os.path.abspath(f"data/v1.1/en/audios_wav.zip")
        def update_zip_paths(example):
            example["zip_path"] = zip_path
            example["internal_path"] = os.path.join("audios", os.path.basename(example["audio_path"]))
            return example
        test_ds = test_ds.map(update_zip_paths)

    def filter_by_length(example):
        s_time = example.get(next((k for k in ["start_time", "start", "segment_start"] if k in example), None))
        e_time = example.get(next((k for k in ["end_time", "end", "segment_end"] if k in example), None))
        try:
            frames = get_audio_frames_from_zip(example["zip_path"], example["internal_path"], start_time=s_time, end_time=e_time, sr=16000)
            return 1000 <= frames <= 480000
        except Exception:
            return False 

    initial_len = len(test_ds)
    if args.dataset == "covost2":
        test_ds = test_ds.filter(filter_by_length, num_proc=4)
        print(f"| filtered out {initial_len - len(test_ds)} (too long/short/corrupted), {len(test_ds)} remained.")

    if args.num_samples is not None:
        test_ds = test_ds.select(range(args.num_samples))

    # Pre-compute LLM prompt components if using the bridge
    if args.baseline == "none":
        st_instruction = ST_PROMPTS[args.tgt_lang]
        model_lower = args.model_name.lower()
        if "qwen" in model_lower:
            prefix_text = f"<|im_start|>system\nYou are a helpful translation assistant.<|im_end|>\n<|im_start|>user\n\"\n"
            postfix_text = f"\"\n{st_instruction}\nOutput only the translation and nothing else.<|im_end|>\n<|im_start|>assistant\n"
        elif "llama" in model_lower:
            prefix_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful translation assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\""
            postfix_text = f"\"\n{st_instruction}\nOutput only the translation and nothing else.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            prefix_text = f"System: You are a helpful translation assistant.\n\nUser: \"\n"
            postfix_text = f"\"\n{st_instruction}\nOutput only the translation and nothing else.\n\nAssistant: "
        
        prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        postfix_ids = tokenizer(postfix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with torch.no_grad():
            prefix_embeds = llm.get_input_embeddings()(prefix_ids)[0]
            postfix_embeds = llm.get_input_embeddings()(postfix_ids)[0]


    predictions, references, sources = [], [], []
    print("\n[START] Batched Inference...")
    
    for i in tqdm(range(0, len(test_ds), args.batch_size)):
        batch = test_ds[i : i + args.batch_size]
        
        if args.dataset == "covost2":
            source_texts = batch["sentence"]
            reference_texts = batch["translation"]
        else:
            source_texts = batch.get("original_speech", [t.get("en", "") for t in batch["transcriptions"]])
            reference_texts = [t.get(args.tgt_lang, "") for t in batch["transcriptions"]]
            
        # Load audio if needed
        audio_arrays = []
        if args.baseline != "text_llm":
            start_key = next((k for k in ["start_time", "start", "segment_start"] if k in batch), None)
            end_key = next((k for k in ["end_time", "end", "segment_end"] if k in batch), None)
            for idx in range(len(batch['zip_path'])):
                s_time = batch[start_key][idx] if start_key else None
                e_time = batch[end_key][idx] if end_key else None
                arr = load_audio_from_zip(batch['zip_path'][idx], batch['internal_path'][idx], start_time=s_time, end_time=e_time, sr=16000)
                audio_arrays.append(arr)
        
        B = len(source_texts)
        pred_texts = []
        
        with torch.no_grad():
            
            # --- BASELINE: TEXT ONLY (Transcripts + LLM) ---
            if args.baseline == "text_llm":
                pred_texts = generate_text_translation(source_texts, args, llm, tokenizer, device)
                
            # --- BASELINE: SEAMLESS END-TO-END ---
            elif args.baseline == "seamless_e2e":
                audio_inputs = processor(audios=audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True).to(audio_device)
                out_tokens = seamless_model.generate(**audio_inputs, tgt_lang=SEAMLESS_LANG_MAP[args.tgt_lang], generate_speech=False)[0]
                pred_texts = processor.batch_decode(out_tokens, skip_special_tokens=True)
                
            # --- BASELINE: CASCADED (Seamless ASR + Text LLM) ---
            elif args.baseline == "cascaded":
                audio_inputs = processor(audios=audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True).to(audio_device)
                # 1. Transcribe to English
                out_tokens = seamless_model.generate(**audio_inputs, tgt_lang="eng", generate_speech=False)[0]
                transcripts = processor.batch_decode(out_tokens, skip_special_tokens=True)
                # 2. Translate with LLM
                pred_texts = generate_text_translation(transcripts, args, llm, tokenizer, device)
                
            else:
                audio_inputs = processor(audios=audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
                audio_inputs = audio_inputs.to(device=audio_device, dtype=torch.bfloat16)
                
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
                    speech_embeds_b = speech_embeds_list[b].to(dtype=torch.bfloat16, device=prefix_embeds.device)
                    seq = torch.cat([prefix_embeds, speech_embeds_b, postfix_embeds], dim=0)
                    batch_inputs_embeds.append(seq)
                    max_len = max(max_len, seq.size(0))
                    
                inputs_embeds = torch.zeros((B, max_len, llm_dim), dtype=torch.bfloat16, device=prefix_embeds.device)
                attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=prefix_embeds.device)
                
                for b in range(B):
                    seq_len = batch_inputs_embeds[b].size(0)
                    inputs_embeds[b, max_len - seq_len:] = batch_inputs_embeds[b]
                    attention_mask[b, max_len - seq_len:] = 1
                    
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)
                
                output_ids = llm.generate(inputs_embeds=inputs_embeds, 
                                          attention_mask=attention_mask, 
                                          max_new_tokens=150, 
                                          num_beams=args.num_beams, 
                                          do_sample=False, 
                                          pad_token_id=tokenizer.eos_token_id)
                pred_texts = [tokenizer.decode(out, skip_special_tokens=True).strip() for out in output_ids]
                
        # Logging batch outputs
        for b in range(B):
            sources.append(source_texts[b])
            references.append(reference_texts[b])
            predictions.append(pred_texts[b])

    predictions = [str(p) if p is not None else "" for p in predictions]
    references = [str(r) if r is not None else "" for r in references]
    sources = [str(s) if s is not None else "" for s in sources]

    print("\nComputing SacreBLEU scores...")
    bleu_scorer = BLEU(lowercase=False, tokenize='13a', smooth_method='exp', effective_order=False)
    bleu_score = bleu_scorer.corpus_score(predictions, [references]).score

    print("Loading COMET model (Unbabel/wmt22-comet-da)...")
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)
    
    print("Computing COMET scores...")
    comet_data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(sources, predictions, references)]
    comet_output = comet_model.predict(comet_data, batch_size=8, gpus=1)
    mean_comet = comet_output.system_score * 100

    print("Computing Language Detection Statistics...")
    lang_stats = {"target": 0, "en": 0, "other": 0, "failed": 0}
    for pred in predictions:
        if not pred.strip():
            lang_stats["failed"] += 1
            continue
        try:
            detected = detect(pred)
            if detected.startswith(args.tgt_lang): lang_stats["target"] += 1
            elif detected == "en": lang_stats["en"] += 1
            else: lang_stats["other"] += 1
        except LangDetectException:
            lang_stats["failed"] += 1

    total_preds = len(predictions)
    lang_percentages = {k: (v / max(1, total_preds)) * 100 for k, v in lang_stats.items()}

    run_type = args.baseline if args.baseline != "none" else "acif"

    print("\n" + "="*50)
    print(f"EVALUATION COMPLETE ({total_preds} Samples)")
    print(f"Task: en -> {args.tgt_lang.upper()} | Dataset: {args.dataset}")
    print(f"Run Type: {run_type.upper()} | LLM: {args.model_name}")
    print(f"COMET Score: {mean_comet:.2f}")
    print(f"BLEU Score : {bleu_score:.2f}")
    print(f"Target Lang: {lang_percentages['target']:.1f}% | English Fallback: {lang_percentages['en']:.1f}%")
    print("="*50)

    model_tag = args.model_name.split("/")[-1]

    if args.baseline != "none":
        save_dir = os.path.join("checkpoints", "baselines", run_type, model_tag)
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = os.path.dirname(args.ckpt_path)


    res_filename = f"{args.split}_st_{args.dataset}_en_{args.tgt_lang}_{run_type}.json"
    res_path = os.path.join(save_dir, res_filename)

    results_dict = {
        "evaluation_config": {
            "dataset": args.dataset,
            "target_language": args.tgt_lang,
            "model_name": args.model_name,
            "run_type": run_type,
            "num_samples": total_preds,
        },
        "metrics": {
            "comet_score": mean_comet,
            "bleu_score": bleu_score,
            "language_detection": {"counts": lang_stats, "percentages": lang_percentages}
        }
    }

    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)
        
    print(f"\nSaved evaluation metrics to: {res_path}")
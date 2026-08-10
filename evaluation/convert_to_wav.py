import os
import glob
import argparse
import librosa
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import warnings

def process_file(file_path, output_dir, target_sr):
    try:
        filename = os.path.basename(file_path)
        name_only = os.path.splitext(filename)[0]
        out_path = os.path.join(output_dir, f"{name_only}.wav")
        
        if os.path.exists(out_path):
            return True
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr, _ = librosa.load(file_path, sr=target_sr, mono=True)
            sf.write(out_path, arr, target_sr, format='WAV')
            
        return True
    except Exception as e:
        print(f"\nFailed to process {file_path}: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input directory containing original audio files")
    parser.add_argument("-o", "--output", required=True, help="Output directory to save the WAV files")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate (default: 16000)")
    parser.add_argument("--cores", type=int, default=16, help="Number of CPU cores to use (default: 16)")
    
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    files = glob.glob(os.path.join(args.input, "*.*"))
    
    if not files:
        print(f"No files found in {args.input}")
        exit()

    print(f"Starting conversion of {len(files)} files to {args.sr}Hz WAV...")

    worker_func = partial(process_file, output_dir=args.output, target_sr=args.sr)
    
    with ProcessPoolExecutor(max_workers=args.cores) as executor:
        list(tqdm(executor.map(worker_func, files), total=len(files)))
        
    print("Conversion Complete!")
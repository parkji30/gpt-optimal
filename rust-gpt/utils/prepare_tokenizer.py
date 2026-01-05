#!/usr/bin/env python3
"""
Download and prepare GPT-2 tokenizer files for use in Rust.

This script downloads the official GPT-2 tokenizer files (vocab.json and merges.txt)
and saves them for use by the Rust tokenizer implementation.
"""

import os
import json
import argparse
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: 'requests' library not found. Install with: pip install requests")

try:
    from transformers import GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: 'transformers' library not found for tokenizer download")


# OpenAI GPT-2 tokenizer URLs
GPT2_VOCAB_URL = "https://huggingface.co/gpt2/raw/main/vocab.json"
GPT2_MERGES_URL = "https://huggingface.co/gpt2/raw/main/merges.txt"


def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL."""
    if not HAS_REQUESTS:
        print(f"Cannot download {url} without 'requests' library")
        return False
    
    try:
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_gpt2_tokenizer_files(output_dir: Path) -> bool:
    """Download GPT-2 vocab.json and merges.txt."""
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"
    
    success = True
    
    if not vocab_path.exists():
        success = download_file(GPT2_VOCAB_URL, vocab_path) and success
    else:
        print(f"vocab.json already exists at {vocab_path}")
    
    if not merges_path.exists():
        success = download_file(GPT2_MERGES_URL, merges_path) and success
    else:
        print(f"merges.txt already exists at {merges_path}")
    
    return success


def download_using_transformers(output_dir: Path) -> bool:
    """Download tokenizer using HuggingFace transformers library."""
    if not HAS_TRANSFORMERS:
        return False
    
    try:
        print("Downloading GPT-2 tokenizer using transformers library...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        # Save vocab.json
        vocab_path = output_dir / "vocab.json"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer.encoder, f, ensure_ascii=False, indent=2)
        print(f"Saved vocab.json to {vocab_path}")
        
        # Save merges.txt
        merges_path = output_dir / "merges.txt"
        with open(merges_path, 'w', encoding='utf-8') as f:
            f.write("#version: 0.2\n")
            for merge in tokenizer.bpe_ranks.keys():
                f.write(f"{merge[0]} {merge[1]}\n")
        print(f"Saved merges.txt to {merges_path}")
        
        # Also save the special tokens
        special_tokens_path = output_dir / "special_tokens.json"
        special_tokens = {
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token": tokenizer.bos_token if tokenizer.bos_token else tokenizer.eos_token,
            "bos_token_id": tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token if tokenizer.pad_token else tokenizer.eos_token,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            "vocab_size": tokenizer.vocab_size,
        }
        with open(special_tokens_path, 'w', encoding='utf-8') as f:
            json.dump(special_tokens, f, indent=2)
        print(f"Saved special_tokens.json to {special_tokens_path}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading tokenizer: {e}")
        return False


def verify_tokenizer(output_dir: Path) -> bool:
    """Verify the downloaded tokenizer files are valid."""
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"
    
    try:
        # Check vocab.json
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        print(f"Vocab size: {len(vocab)}")
        
        # Check merges.txt
        with open(merges_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Skip header line if present
        num_merges = len([l for l in lines if l.strip() and not l.startswith('#')])
        print(f"Number of merges: {num_merges}")
        
        return len(vocab) > 0 and num_merges > 0
        
    except Exception as e:
        print(f"Error verifying tokenizer: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download GPT-2 tokenizer files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/tokenizer",
        help="Output directory for tokenizer files"
    )
    parser.add_argument(
        "--use-transformers",
        action="store_true",
        help="Use transformers library to download (recommended)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading GPT-2 tokenizer to {output_dir.absolute()}")
    print("="*50)
    
    success = False
    
    if args.use_transformers or HAS_TRANSFORMERS:
        success = download_using_transformers(output_dir)
    
    if not success:
        print("\nFalling back to direct download...")
        success = download_gpt2_tokenizer_files(output_dir)
    
    if success:
        print("\n" + "="*50)
        print("Verifying tokenizer files...")
        if verify_tokenizer(output_dir):
            print("✓ Tokenizer files are valid!")
        else:
            print("✗ Tokenizer verification failed!")
    else:
        print("\n✗ Failed to download tokenizer files")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())



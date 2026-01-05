#!/usr/bin/env python3
"""
Download and prepare the Nemotron dataset for GPT-2 training.

The Nemotron datasets from NVIDIA include:
- nvidia/Nemotron-4-340B-Instruct
- nvidia/HelpSteer2
- Various synthetic data generation datasets

This script downloads a subset suitable for training a small GPT-2 model.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional

# Try to import huggingface datasets
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' library not found. Install with: pip install datasets")

# Try to import requests for fallback download
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def download_helpsteer2(output_dir: Path, max_samples: Optional[int] = None) -> int:
    """
    Download the HelpSteer2 dataset from HuggingFace.
    This is a high-quality dataset used for training Nemotron models.
    """
    if not HAS_DATASETS:
        print("Cannot download from HuggingFace without 'datasets' library")
        return 0
    
    print("Downloading HelpSteer2 dataset from HuggingFace...")
    
    try:
        dataset = load_dataset("nvidia/HelpSteer2", split="train")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # Save as JSONL for easy loading in Rust
        output_file = output_dir / "helpsteer2_train.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                # Extract text content - combine prompt and response
                text = f"Human: {item.get('prompt', '')}\n\nAssistant: {item.get('response', '')}"
                json.dump({"text": text, "source": "helpsteer2"}, f)
                f.write('\n')
        
        print(f"Saved {len(dataset)} samples to {output_file}")
        return len(dataset)
        
    except Exception as e:
        print(f"Error downloading HelpSteer2: {e}")
        return 0


def download_openwebtext_sample(output_dir: Path, max_samples: Optional[int] = 10000) -> int:
    """
    Download a sample from OpenWebText dataset as fallback/additional training data.
    """
    if not HAS_DATASETS:
        print("Cannot download from HuggingFace without 'datasets' library")
        return 0
    
    print("Downloading OpenWebText sample...")
    
    try:
        # Load a streaming subset to avoid downloading the full dataset
        dataset = load_dataset("openwebtext", split="train", streaming=True)
        
        output_file = output_dir / "openwebtext_sample.jsonl"
        count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                text = item.get('text', '')
                if len(text) > 100:  # Filter very short texts
                    json.dump({"text": text, "source": "openwebtext"}, f)
                    f.write('\n')
                    count += 1
                    
                    if count % 1000 == 0:
                        print(f"Downloaded {count} samples...")
                    
                    if max_samples and count >= max_samples:
                        break
        
        print(f"Saved {count} samples to {output_file}")
        return count
        
    except Exception as e:
        print(f"Error downloading OpenWebText: {e}")
        return 0


def download_wikitext(output_dir: Path) -> int:
    """
    Download WikiText-2 dataset - good for language modeling benchmarks.
    """
    if not HAS_DATASETS:
        print("Cannot download from HuggingFace without 'datasets' library")
        return 0
    
    print("Downloading WikiText-2 dataset...")
    
    try:
        dataset = load_dataset("wikitext", "wikitext-2-v1")
        
        total_samples = 0
        
        for split in ['train', 'validation', 'test']:
            output_file = output_dir / f"wikitext2_{split}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in dataset[split]:
                    text = item.get('text', '').strip()
                    if text:  # Skip empty lines
                        json.dump({"text": text, "source": "wikitext2"}, f)
                        f.write('\n')
                        total_samples += 1
            
            print(f"Saved {split} split to {output_file}")
        
        print(f"Total WikiText-2 samples: {total_samples}")
        return total_samples
        
    except Exception as e:
        print(f"Error downloading WikiText: {e}")
        return 0


def create_demo_dataset(output_dir: Path, num_samples: int = 1000) -> int:
    """
    Create a demo dataset with synthetic text for testing.
    """
    print(f"Creating demo dataset with {num_samples} samples...")
    
    templates = [
        "The {} is a {} that {} in the {}.",
        "Scientists discovered that {} can {} when exposed to {}.",
        "In the year {}, {} will become {} due to {}.",
        "The art of {} requires {} and {} to master.",
        "When {} meets {}, the result is always {}.",
        "The theory of {} explains how {} interacts with {}.",
        "Every {} contains {} that help {} function properly.",
        "The history of {} shows us that {} is important for {}.",
        "Modern {} uses {} to achieve {} efficiency.",
        "The relationship between {} and {} determines the {}.",
    ]
    
    nouns = ["machine", "algorithm", "network", "system", "model", "process", 
             "structure", "pattern", "method", "approach", "technique", "framework",
             "computer", "program", "language", "database", "interface", "protocol"]
    
    adjectives = ["innovative", "efficient", "complex", "simple", "advanced",
                  "traditional", "modern", "classical", "dynamic", "static"]
    
    verbs = ["transforms", "processes", "analyzes", "generates", "optimizes",
             "implements", "develops", "creates", "manages", "controls"]
    
    import random
    random.seed(42)
    
    output_file = output_dir / "demo_dataset.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            template = random.choice(templates)
            # Count placeholders
            num_placeholders = template.count("{}")
            
            words = []
            for _ in range(num_placeholders):
                word_type = random.choice(['noun', 'adj', 'verb'])
                if word_type == 'noun':
                    words.append(random.choice(nouns))
                elif word_type == 'adj':
                    words.append(random.choice(adjectives))
                else:
                    words.append(random.choice(verbs))
            
            text = template.format(*words)
            json.dump({"text": text, "source": "demo", "id": i}, f)
            f.write('\n')
    
    print(f"Saved {num_samples} demo samples to {output_file}")
    return num_samples


def combine_datasets(output_dir: Path, output_name: str = "combined_train.jsonl"):
    """
    Combine all downloaded datasets into a single training file.
    """
    output_file = output_dir / output_name
    total = 0
    
    print(f"Combining datasets into {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as outf:
        for jsonl_file in output_dir.glob("*.jsonl"):
            if jsonl_file.name == output_name:
                continue
            
            print(f"  Adding {jsonl_file.name}...")
            with open(jsonl_file, 'r', encoding='utf-8') as inf:
                for line in inf:
                    outf.write(line)
                    total += 1
    
    print(f"Combined {total} samples into {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare training datasets")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["demo", "wikitext"],
        choices=["helpsteer2", "openwebtext", "wikitext", "demo", "all"],
        help="Which datasets to download"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (for testing)"
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all datasets into one file"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = ["helpsteer2", "openwebtext", "wikitext", "demo"]
    
    total_samples = 0
    
    for dataset in datasets_to_download:
        print(f"\n{'='*50}")
        print(f"Processing: {dataset}")
        print('='*50)
        
        if dataset == "helpsteer2":
            total_samples += download_helpsteer2(output_dir, args.max_samples)
        elif dataset == "openwebtext":
            total_samples += download_openwebtext_sample(output_dir, args.max_samples or 10000)
        elif dataset == "wikitext":
            total_samples += download_wikitext(output_dir)
        elif dataset == "demo":
            total_samples += create_demo_dataset(output_dir, args.max_samples or 1000)
    
    if args.combine:
        combine_datasets(output_dir)
    
    print(f"\n{'='*50}")
    print(f"Download complete! Total samples: {total_samples}")
    print(f"Data saved to: {output_dir.absolute()}")
    print('='*50)


if __name__ == "__main__":
    main()



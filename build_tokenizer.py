#!/usr/bin/env python3
"""
Build a custom BPE tokenizer from the Gutenberg dataset using SentencePiece.
Produces: kahler_tokenizer.model + kahler_tokenizer.vocab

Usage:
    python build_tokenizer.py                    # default vocab=4096
    python build_tokenizer.py --vocab-size 8192  # custom size
"""

import argparse
import os
import tempfile
import sentencepiece as spm
from datasets import load_dataset


def build_tokenizer(vocab_size=4096, model_prefix="kahler_tokenizer"):
    print(f"[1/3] Loading Gutenberg dataset...")
    ds = load_dataset("deven367/babylm-100M-gutenberg", split="train")
    print(f"       {len(ds):,} rows loaded")

    # Export to temp text file for SentencePiece
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gutenberg_corpus.txt")
    print(f"[2/3] Exporting corpus to {tmp_path}...")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"].strip()
            if text:
                f.write(text + "\n")
    corpus_size = os.path.getsize(tmp_path)
    print(f"       Corpus size: {corpus_size / 1e6:.1f} MB")

    # Train SentencePiece BPE
    print(f"[3/3] Training BPE tokenizer (vocab_size={vocab_size})...")
    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,       # full coverage for English
        pad_id=3,                      # reserve 0=<unk>, 1=<s>, 2=</s>, 3=<pad>
        unk_id=0,
        bos_id=1,
        eos_id=2,
        normalization_rule_name="identity",  # no NFKC — keep original text
        byte_fallback=True,            # handle any unseen byte
        num_threads=os.cpu_count(),
    )

    # Clean up temp file
    os.remove(tmp_path)
    print(f"\n✅ Tokenizer saved:")
    print(f"   Model: {model_prefix}.model")
    print(f"   Vocab: {model_prefix}.vocab")

    # Quick verification
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    test_sentences = [
        "Mary opens the door at night",
        "The light in the library was dim and cold.",
        "She looked at him and said nothing.",
        "Once upon a time there was a little girl.",
    ]
    print(f"\n📋 Verification (vocab_size={sp.get_piece_size()}):")
    for sent in test_sentences:
        tokens = sp.encode(sent, out_type=str)
        ids = sp.encode(sent)
        decoded = sp.decode(ids)
        roundtrip_ok = "✅" if decoded == sent else "❌"
        print(f"   {roundtrip_ok} \"{sent}\"")
        print(f"      → {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        print(f"      → {len(tokens)} tokens, roundtrip: \"{decoded}\"")

    print(f"\n📊 Token stats:")
    print(f"   Vocab size: {sp.get_piece_size()}")
    print(f"   <unk>=0, <s>=1, </s>=2, <pad>=3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build custom BPE tokenizer")
    parser.add_argument("--vocab-size", type=int, default=4096,
                        help="Vocabulary size (default: 4096)")
    parser.add_argument("--model-prefix", type=str, default="kahler_tokenizer",
                        help="Output model prefix (default: kahler_tokenizer)")
    args = parser.parse_args()
    build_tokenizer(vocab_size=args.vocab_size, model_prefix=args.model_prefix)

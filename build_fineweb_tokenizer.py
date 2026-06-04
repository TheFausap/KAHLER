import os
os.environ["HF_HOME"] = "/home/fausap/Documents/LLM/CACHE/huggingface"

import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm

CORPUS_FILE = "fineweb_corpus.txt"
MODEL_PREFIX = "fineweb_tokenizer"
VOCAB_SIZE = 16384
TARGET_BYTES = 100 * 1024 * 1024  # ~100 MB of text

def build_corpus():
    print("Streaming HuggingFaceFW/fineweb-edu (sample-10BT)...")
    dataset = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
    
    bytes_written = 0
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        with tqdm(total=TARGET_BYTES, unit="B", unit_scale=True, desc="Collecting text") as pbar:
            for item in dataset:
                text = item['text'].replace('\n', ' ') + '\n'
                f.write(text)
                written = len(text.encode('utf-8'))
                bytes_written += written
                pbar.update(written)
                
                if bytes_written >= TARGET_BYTES:
                    break

def train_tokenizer():
    print(f"\nTraining SentencePiece model with vocab_size={VOCAB_SIZE}...")
    spm.SentencePieceTrainer.train(
        input=CORPUS_FILE,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type='bpe',
        character_coverage=0.9995,
        pad_id=3,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
        user_defined_symbols=['<|endoftext|>'],
        max_sentence_length=10000,
        shuffle_input_sentence=True
    )
    print(f"[+] Tokenizer trained and saved as {MODEL_PREFIX}.model")

if __name__ == "__main__":
    if not os.path.exists(CORPUS_FILE):
        build_corpus()
    else:
        print(f"Corpus {CORPUS_FILE} already exists, skipping collection.")
    train_tokenizer()

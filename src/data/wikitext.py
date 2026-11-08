"""
WikiText-103 data loader for training and extraction.

Produces tokenized chunks of length L=2048 for next-token prediction.
Uses GPT-2 tokenizer (vocab_size=50257).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast


class WikiTextDataset(Dataset):
    """Pre-tokenized WikiText-103 chunks of fixed length."""

    def __init__(self, split: str = 'train', seq_len: int = 2048):
        self.seq_len = seq_len

        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        raw = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)

        # Tokenize in batches to avoid OOM from one giant string
        chunk_size = seq_len + 1
        all_tokens = []
        batch_lines = []
        batch_char_count = 0
        # Process ~10MB of text at a time
        char_limit = 10_000_000

        for line in raw['text']:
            if not line.strip():
                continue
            batch_lines.append(line)
            batch_char_count += len(line)

            if batch_char_count >= char_limit:
                text = '\n'.join(batch_lines)
                ids = tokenizer.encode(text)
                all_tokens.extend(ids)
                batch_lines = []
                batch_char_count = 0

        # Flush remaining
        if batch_lines:
            text = '\n'.join(batch_lines)
            ids = tokenizer.encode(text)
            all_tokens.extend(ids)

        tokens = torch.tensor(all_tokens, dtype=torch.long)

        # Chunk into sequences of seq_len + 1 (input + target)
        n_chunks = len(tokens) // chunk_size
        tokens = tokens[:n_chunks * chunk_size]
        self.chunks = tokens.view(n_chunks, chunk_size)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]  # input_ids, target_ids


def get_dataloader(
    split: str = 'train',
    seq_len: int = 2048,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    dataset = WikiTextDataset(split=split, seq_len=seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

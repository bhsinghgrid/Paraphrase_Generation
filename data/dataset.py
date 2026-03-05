from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import random
from config import CONFIG


class OptimizedSanskritDataset(Dataset):
    def __init__(self, split='train', tokenizer=None, max_len=80):
        print(f"📥 Loading '{split}' split of the dataset...")
        self.dataset = load_dataset("paws/sanskrit-verses-gretil", split=split)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.include_negatives = CONFIG['data']['include_negative_examples']

        if split == 'train':
            self.dataset = self._curriculum_sort()
        print(f"✅ Dataset loaded! Total samples: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def _curriculum_sort(self):
        difficulties = []
        for sample in self.dataset:
            length = len(sample['quote_text'].split())
            rarity_score = len(set(sample['quote_devanagari'])) / max(1, len(sample['quote_devanagari']))
            difficulties.append(length * (1 - rarity_score))
        order = sorted(range(len(self.dataset)), key=lambda i: difficulties[i])
        return self.dataset.select(order)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        input_ids = torch.tensor(self.tokenizer.encode(sample['quote_text'])[:self.max_len], dtype=torch.long)
        target_ids = torch.tensor(self.tokenizer.encode(sample['quote_devanagari'])[:self.max_len], dtype=torch.long)

        input_ids = F.pad(input_ids, (0, max(0, self.max_len - len(input_ids))), value=1)  # 1 is PAD
        target_ids = F.pad(target_ids, (0, max(0, self.max_len - len(target_ids))), value=1)

        data_dict = {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'input_text': sample['quote_text'],
            'target_text': sample['quote_devanagari']
        }

        # Inject negative example (shuffled target) if requested
        if self.include_negatives:
            neg_ids = target_ids.clone()
            # Shuffle a random chunk to create a grammatically incorrect negative
            if len(neg_ids) > 4:
                idx1, idx2 = sorted(random.sample(range(len(neg_ids)), 2))
                neg_ids[idx1:idx2] = torch.flip(neg_ids[idx1:idx2], dims=[0])
            data_dict['negative_target_ids'] = neg_ids

        return data_dict

    def compute_frequencies(self):
        print("📊 Computing Token Frequencies for Importance Sampling...")
        counts = torch.zeros(self.tokenizer.vocab_size)

        # Sample 10% of the dataset for speed, or use the whole thing for precision
        sample_size = min(50000, len(self.data))
        for i in range(sample_size):
            tokens = self.tokenizer.encode(self.data[i]['sanskrit'])
            for t in tokens:
                if t < self.tokenizer.vocab_size:
                    counts[t] += 1

        # Normalize: rare tokens get higher "Importance" (closer to 1.0)
        # common tokens get lower "Importance" (closer to 0.1)
        freqs = counts / (counts.max() + 1e-9)
        importance = 1.0 - (freqs * 0.9)

        return importance.to(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
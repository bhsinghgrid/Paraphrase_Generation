from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import random

class OptimizedSanskritDataset(Dataset):
    """
    📚 Custom Dataset for Sanskrit Verse Translation with optional negative examples
    Handles IAST → Devanagari conversion with tokenization, curriculum sorting, and negative examples.
    """
    def __init__(self, split='train', tokenizer=None, max_len=80, include_negative=False):
        print(f"📥 Loading '{split}' split of the dataset...")

        self.dataset = load_dataset("paws/sanskrit-verses-gretil", split=split)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.include_negative = include_negative

        if split == 'train':
            self.dataset = self._curriculum_sort()

        # If negative examples are included, create a pool of target texts
        if self.include_negative:
            self.target_pool = [sample['quote_devanagari'] for sample in self.dataset]

        print(f"✅ Dataset loaded successfully! Total samples: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def _curriculum_sort(self):
        """
        🎯 Curriculum Learning:
        Sort samples from easy → hard based on length and character rarity.
        """
        difficulties = []

        for sample in self.dataset:
            length = len(sample['quote_text'].split())
            rarity_score = len(set(sample['quote_devanagari'])) / len(sample['quote_devanagari'])
            difficulties.append(length * (1 - rarity_score))

        order = sorted(range(len(self.dataset)), key=lambda i: difficulties[i])
        return self.dataset.select(order)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Decide if this will be a negative example
        if self.include_negative and random.random() < 0.5:
            # Pick a random target that does NOT match this input
            negative_target = sample['quote_devanagari']
            while negative_target == sample['quote_devanagari']:
                negative_target = random.choice(self.target_pool)
            target_text = negative_target
            is_negative = True
        else:
            target_text = sample['quote_devanagari']
            is_negative = False

        # Tokenize input and target
        input_ids = torch.tensor(self.tokenizer.encode(sample['quote_text'])[:self.max_len], dtype=torch.long)
        target_ids = torch.tensor(self.tokenizer.encode(target_text)[:self.max_len], dtype=torch.long)

        # Pad sequences
        input_ids = F.pad(input_ids, (0, max(0, self.max_len - len(input_ids))), value=0)
        target_ids = F.pad(target_ids, (0, max(0, self.max_len - len(target_ids))), value=0)

        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'input_text': sample['quote_text'],
            'target_text': target_text,
            'is_negative': is_negative   # ✅ Flag to indicate negative sample
        }
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

class OptimizedSanskritDataset(Dataset):
    """
    📚 Custom Dataset for Sanskrit Verse Translation
    Handles IAST → Devanagari conversion with tokenization and curriculum sorting.
    """
    def __init__(self, split='train', tokenizer=None, max_len=80):
        # Announce which split is being loaded
        print(f"📥 Loading '{split}' split of the dataset...")

        # Load dataset from HuggingFace hub
        self.dataset = load_dataset("paws/sanskrit-verses-gretil", split=split)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Apply curriculum learning (easy → hard) for training split
        if split == 'train':
            self.dataset = self._curriculum_sort()

        # Confirm load
        print(f"✅ Dataset loaded successfully! Total samples: {len(self.dataset)}")

    def __len__(self):
        # Return number of samples in the dataset
        return len(self.dataset)

    def _curriculum_sort(self):
        """
        🎯 Curriculum Learning:
        Sort samples from easy → hard based on length and character rarity.
        """
        difficulties = []

        for sample in self.dataset:
            # Longer sentences are harder
            length = len(sample['quote_text'].split())

            # Rare characters increase difficulty
            rarity_score = len(set(sample['quote_devanagari'])) / len(sample['quote_devanagari'])

            # Combine length and rarity into a single difficulty metric
            difficulties.append(length * (1 - rarity_score))

        # Get indices sorted by difficulty
        order = sorted(range(len(self.dataset)), key=lambda i: difficulties[i])

        # Reorder dataset
        return self.dataset.select(order)

    def __getitem__(self, idx):
        """
        🔥 Return tokenized input and target with padding
        Always returns a dict containing:
        - input_ids
        - target_ids
        - input_text (original IAST)
        - target_text (original Devanagari)
        """
        sample = self.dataset[idx]

        # Tokenize input (IAST) and target (Devanagari)
        input_ids = torch.tensor(
            self.tokenizer.encode(sample['quote_text'])[:self.max_len],
            dtype=torch.long
        )
        target_ids = torch.tensor(
            self.tokenizer.encode(sample['quote_devanagari'])[:self.max_len],
            dtype=torch.long
        )

        # Pad sequences to max_len
        input_ids = F.pad(input_ids, (0, max(0, self.max_len - len(input_ids))), value=0)
        target_ids = F.pad(target_ids, (0, max(0, self.max_len - len(target_ids))), value=0)

        # Return dictionary
        return {
            'input_ids': input_ids,       # 🔑 Required by model
            'target_ids': target_ids,     # 🔑 Required by model
            'input_text': sample['quote_text'],
            'target_text': sample['quote_devanagari']
        }
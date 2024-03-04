from torch import tensor
from torch.utils.data import Dataset, DataLoader


class DatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_len, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_len, stride):
            input_chunk = token_ids[i:i + max_len]
            target_chunk = token_ids[i + 1:i + max_len + 1]
            self.input_ids.append(tensor(input_chunk))
            self.target_ids.append(tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_loader(
    txt,
    tokenizer,
    batch_size=4,
    max_len=256,
    stride=128,
    shuffle=True,
):
    dataset = DatasetV1(txt, tokenizer, max_len, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

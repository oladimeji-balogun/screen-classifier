from sklearn.model_selection import train_test_split 
from torch.utils.data import Dataset

def get_train_val_indices(dataset: Dataset, val_ratio: float = 0.15, seed = 20):
    indices = range(len(dataset))
    labels = [dataset[idx][1] for idx in indices] 
    train_indices, val_indices = train_test_split(indices, random_state=seed, test_size=val_ratio, stratify=labels)
    return train_indices, val_indices


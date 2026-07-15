from torch.utils.data import Dataset 
from pathlib import Path
from PIL import Image

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

class SceneDataset(Dataset):
    def __init__(self, root_path, split, transform=None): 
        if split == "train": 
            self.path = Path(root_path) / "seg_train" / "seg_train"
        elif split == "test": 
            self.path = Path(root_path) / "seg_test" / "seg_test"
        else: 
            raise ValueError("invalid split directory")

        self.transform = transform
        self.images = []

        for item in Path(self.path).rglob('*'): 
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']: 
                self.images.append(
                    (item, item.parent.name.lower())
                )

    def __len__(self): 
        return len(self.images)
    
    def __getitem__(self, index):
        image_path, label = self.images[index]
        image = Image.open(image_path).convert("RGB")
        label_idx = CLASS_NAMES.index(label)

        if self.transform is not None: 
            return self.transform(image), label_idx
        else: 
            return image, label_idx
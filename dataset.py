import clip
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, data_path: str, model_name: str, image_transform, ruclip_processor=None):
        super().__init__()

        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        self.model_name = model_name
        self.image_transform = image_transform
        self.ruclip_processor = ruclip_processor

        if model_name == 'clip':
            self.texts = clip.tokenize(self.data['text'], truncate=True)
        elif model_name == 'ruclip':
            if ruclip_processor is not None:
                self.texts = ruclip_processor(text=self.data['text'], return_tensors='pt', padding=True, truncation=True)['input_ids']
            else:
                raise Exception('Pass processor for ruCLIP!')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.image_transform(Image.open(self.data.loc[index, 'local_image_path']))
        text = self.texts[index]
        return image, text
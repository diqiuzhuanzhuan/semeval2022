import pytorch_lightning as pl
import transformers
from typing import Optional
import torch.utils.data
from utils.reader import conll_reader
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import LineByLineTextDataset


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')

class MoonRobertaDataModule(pl.LightningDataModule):
    
    def __init__(self, train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms, dims=dims)

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        return super().setup(stage=stage)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return super().train_dataloader()

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return super().test_dataloader()

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return super().val_dataloader()

    def teardown(self, stage: Optional[str] = None) -> None:
        return super().teardown(stage=stage)

    

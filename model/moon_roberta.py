import pytorch_lightning as pl
import transformers
from typing import Optional
import torch.utils.data
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

    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x1 = [1, 2, 3, 4, 5]
    y1 = [0.636, 0.880, 0.900, 0.891, 0.906]
    x2 = [1, 2, 3, 4, 5, 6, 7, 8]
    y2 = [0.859, 0.876, 0.897, 0.893, 0.901, 0.891, 0.893, 0.902]
    l1=plt.plot(x1,y1,'r--',label='with auxiliary task')
    l2=plt.plot(x2,y2,'g--',label='without auxiliary task')
    plt.plot(x1, y1, 'ro-', x2, y2, 'g+-')
    plt.title('comparison of convergence speed')
    plt.xlabel('epoch')
    plt.ylabel('macro f1')
    plt.legend()
    plt.show()
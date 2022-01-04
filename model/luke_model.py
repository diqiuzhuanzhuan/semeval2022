from re import I
from typing import List, Any
import numpy as np
from numpy.core.fromnumeric import argmax, sort
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from transformers.pipelines import token_classification
from transformers.utils.dummy_pt_objects import LukeForEntitySpanClassification
from transformers import *
import torch
from utils.conll_reader import LukeCoNLLReader
from utils.metric import SpanF1
from intervaltree import Interval, IntervalTree

def negative_sample(label: torch.Tensor, sample_length: List[int], max_negative_positive_radio: 3, negative_label):
    batch_size, label_len = label.size()[0], label.size()[1]
    sample_index = []
    labels_tensor = label.clone().detach()
    for i in range(batch_size):
        actual_label = label[i][0:sample_length[i]]
        index = (actual_label == negative_label).nonzero(as_tuple=True)[0]
        positive_sample_index = (actual_label != negative_label).nonzero(as_tuple=True)[0].tolist()
        positive_sample = len(positive_sample_index)
        max_negative_sample = max(min(round(positive_sample * max_negative_positive_radio), sample_length[i] - positive_sample), 5)
        sample_negative_index = torch.randperm(len(index))[:max_negative_sample].tolist()
        sample_negative_index.extend(positive_sample_index)
        sample_index.append(sample_negative_index)
        if len(sample_negative_index) == 0:
            print(positive_sample)
            pass
        labels_tensor[i][sample_negative_index] = labels_tensor[i][sample_negative_index]
        

    return sample_index, labels_tensor

class LukeNer(pl.LightningModule):
    
    def __init__(self,
                 encoder_model="studio-ousia/luke-base",
                 batch_size=16, 
                 max_length=100,
                 lr=2e-5,
                 dropout_rate=0.1,
                 train_data: LukeCoNLLReader=None,
                 dev_data: LukeCoNLLReader=None,
                 tag_to_id=None,
                 negative_sample=True,
                 stage='fit'
                 ):
        super(LukeNer, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.train_data = train_data
        self.dev_data = dev_data
        self.max_length = max_length
        self.label_to_id = tag_to_id
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.stage = stage
        self.num_labels = len(self.label_to_id)
        self.encoder = LukeForEntitySpanClassification.from_pretrained(encoder_model, num_labels=self.num_labels)
        self.tokenizer = LukeTokenizer.from_pretrained(encoder_model, task="entity_span_classification")
        self.train_span_f1 = SpanF1()
        self.val_span_f1 = SpanF1()
        self.negative_sample = negative_sample

        uncased_entity_vocab = dict()
        for k in self.tokenizer.entity_vocab:
            uncased_entity_vocab[str.lower(k)] = self.tokenizer.entity_vocab[k]
        for k in uncased_entity_vocab:
            self.tokenizer.entity_vocab[k] = uncased_entity_vocab[k]
    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        return optimizer

    def collate_batch(self, mode='val'):
        if mode == 'fit':
            return self.train_collate_batch
        else:
            return self.val_collate_batch
    
    def train_collate_batch(self, batch):
        batch_ = list(zip(*batch))
        sentence_str, entity_spans, labels, tokens, ner_tags = batch_[0], batch_[1], batch_[2], batch_[3], batch_[4]
        batch_size = len(sentence_str)
        max_label_len = max([len(token) for token in labels])
        labels_tensor = torch.empty(size=(batch_size, max_label_len), dtype=torch.long).fill_(-100)
     
        for i in range(batch_size):
            length = len(labels[i])
            labels_tensor[i, :length] = torch.tensor(labels[i])
        outputs = self.tokenizer(sentence_str, entity_spans=entity_spans, return_tensors="pt", max_length=self.max_length, padding=True, max_entity_length=max_label_len, truncation=True)
        if self.negative_sample:
            label_index, labels_tensor = negative_sample(label=labels_tensor, sample_length=[len(_) for _ in entity_spans], max_negative_positive_radio=0.3, negative_label=self.label_to_id['O'])
            sample_entity_spans = []
            for i in range(batch_size):
                _entity_spans = [entity_spans[i][j] for j in label_index[i]]
                sample_entity_spans.append(_entity_spans)
            entity_spans = sample_entity_spans
        
        
        outputs["labels"] = labels_tensor
        return outputs, tokens, ner_tags, entity_spans, sentence_str

    def val_collate_batch(self, batch):
        batch_ = list(zip(*batch))
        sentence_str, entity_spans, labels, tokens, ner_tags = batch_[0], batch_[1], batch_[2], batch_[3], batch_[4]
        batch_size = len(sentence_str)
        max_label_len = max([len(token) for token in labels])
        labels_tensor = torch.empty(size=(batch_size, max_label_len), dtype=torch.long).fill_(-100)
     
        for i in range(batch_size):
            length = len(labels[i])
            labels_tensor[i, :length] = torch.tensor(labels[i])
        outputs = self.tokenizer(sentence_str, entity_spans=entity_spans, return_tensors="pt", max_length=self.max_length, padding=True, max_entity_length=max_label_len, truncation=True)
        outputs["labels"] = labels_tensor
        return outputs, tokens, ner_tags, entity_spans, sentence_str

    def train_dataloader(self):
        loader = DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_batch(mode='fit'), num_workers=1)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.dev_data, batch_size=self.batch_size, collate_fn=self.collate_batch(mode='val'), num_workers=1)
        return loader

    def training_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch, mode='fit')
        self.log_metrics(output['results'], loss=output['loss'], suffix='', on_step=True, on_epoch=False)
        return output

    def validation_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch, mode='val')
        self.log_metrics(output['results'], loss=output['loss'], suffix='', on_step=True, on_epoch=False)
        return output
    
    def test_epoch_end(self, outputs):
        pred_results = self.val_span_f1.get_metric()
        avg_loss = np.mean([preds['loss'].item() for preds in outputs])
        self.log_metrics(pred_results, loss=avg_loss, on_step=False, on_epoch=True)

        out = {"test_loss": avg_loss, "results": pred_results}
        return out

    def training_epoch_end(self, outputs: List[Any]) -> None:
        pred_results = self.train_span_f1.get_metric(True)
        avg_loss = np.mean([preds['loss'].item() for preds in outputs])
        self.log_metrics(pred_results, loss=avg_loss, suffix='', on_step=False, on_epoch=True)
        self.train_span_f1.reset()

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        pred_results = self.val_span_f1.get_metric(True)
        avg_loss = np.mean([preds['loss'].item() for preds in outputs])
        self.log_metrics(pred_results, loss=avg_loss, suffix='val_', on_step=False, on_epoch=True)
        self.val_span_f1.reset()

    def log_metrics(self, pred_results, loss=0.0, suffix='', on_step=False, on_epoch=True):
        for key in pred_results:
            self.log(suffix + key, pred_results[key], on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)
        self.log(suffix + 'loss', loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

    def perform_forward_step(self, batch, mode=''):
        inputs, tokens, ner_tags, entity_spans, sentence_str = batch
        outputs = self.encoder(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(outputs.logits.view(-1, self.num_labels), inputs["labels"].view(-1))
        print(loss, outputs["loss"])
        outputs["loss"] = loss
        batch_size = len(entity_spans)
        batch_final_res = []
        y_true = []
        y_pred = []
        for i in range(batch_size):
            sentence = sentence_str[i]
            spans = entity_spans[i]
            best = outputs.logits[i].max(dim=-1)
            best_score = best.values.tolist()
            best_indices = best.indices.tolist()
            predictions = sorted([(score, index, span) for score, index, span in zip(best_score, best_indices, spans)], key=lambda k: k[0], reverse=True)
            interval_tree = IntervalTree()
            predict_res = []
            for _, predict_id, span in predictions:
                if interval_tree.overlap(span[0], span[1]):
                    continue
                predict_res.append((span, self.id_to_label[predict_id]))
                interval_tree.add(Interval(span[0], span[1]))
            final_res = []
            for span, label in sorted(predict_res, key=lambda k: k[0][0]):
                entity_words = sentence[span[0]: span[1]].split(" ")
                for idx, word in enumerate(entity_words):
                    if label.startswith("O"):
                        final_res.append((word, label))
                        continue
                    if idx == 0:
                        final_res.append((word, 'B-'+label))
                    else:
                        final_res.append((word, 'I-'+label))
            new_final_res = []
            for k in range(len(tokens[i])):
                j = 0
                if tokens[i][k] != final_res[j][0]:
                    new_final_res.append((tokens[i][k], "O"))
                else:
                    new_final_res.append(final_res[j])
                    j += 1
            final_res = new_final_res
            assert(sentence == " ".join([_[0] for _ in final_res]))
            assert(len(tokens[i]) == len(final_res))
            batch_final_res.append(final_res)
            y_true.append({(i, i+1): tag for i, tag in enumerate(ner_tags[i])})
            y_pred.append({(i, i+1): tag[1] for i, tag in enumerate(final_res)})
        if mode == 'val':
            self.val_span_f1(y_true, y_pred)
            outputs["results"] = self.val_span_f1.get_metric()
        elif mode == 'fit' or mode == 'train':
            self.train_span_f1(y_true, y_pred)
            outputs["results"] = self.train_span_f1.get_metric()
        
        return outputs
    
if  __name__ == "__main__":
    from utils.util import get_reader, train_model, create_model, save_model, parse_args, get_tagset, wnut_iob, write_submit_result, load_model, luke_iob
    import time, os
    base_dir = "."
    encoder_model = "studio-ousia/luke-base"
    track = "EN-English/en"
    train_file = os.path.join(base_dir, "training_data/{}_train.conll".format(track))
    dev_file = os.path.join(base_dir, "training_data/{}_dev.conll".format(track))
    output_dir = os.path.join(base_dir, "{}".format(track), "{}-train".format(encoder_model))
    submission_file = os.path.join(base_dir, "submission", "{}.pred.conll".format(track))
    iob_tagging = luke_iob
    use_crf = True
    train_data = get_reader(file_path=dev_file, target_vocab=iob_tagging, encoder_model=encoder_model, max_instances=32, max_length=100)
    dev_data = get_reader(file_path=dev_file, target_vocab=iob_tagging, encoder_model=encoder_model, max_instances=32, max_length=100)

    model = create_model(train_data=train_data, dev_data=train_data, tag_to_id=iob_tagging,
                     dropout_rate=0.1, batch_size=16, stage='fit', lr=2e-5,
                     encoder_model=encoder_model, num_gpus=1, use_crf=False)

    trainer = train_model(model=model, out_dir=output_dir, epochs=20, monitor="f1")
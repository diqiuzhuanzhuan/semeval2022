from re import I, M
import re
from typing import List, Any

import pytorch_lightning.core.lightning as pl
from pytorch_lightning.utilities.distributed import init_dist_connection
from torch.nn import CrossEntropyLoss
import torch
import torch.nn.functional as F
import numpy as np

from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoModel
from transformers.models import trocr
from transformers.utils.dummy_pt_objects import RobertaForTokenClassification
from transformers import *
from transformers.utils.dummy_tf_objects import WarmUp

from log import logger
from utils.metric import SpanF1
from utils.reader_utils import extract_spans, get_tags


class NERBaseAnnotator(pl.LightningModule):
    def __init__(self,
                 train_data=None,
                 dev_data=None,
                 lr=1e-5,
                 dropout_rate=0.1,
                 batch_size=16,
                 tag_to_id=None,
                 stage='fit',
                 pad_token_id=1,
                 encoder_model='xlm-roberta-large',
                 num_gpus=1,
                 use_crf=False):
        super(NERBaseAnnotator, self).__init__()

        self.train_data = train_data
        self.dev_data = dev_data

        self.id_to_tag = {v: k for k, v in tag_to_id.items()}
        self.tag_to_id = tag_to_id
        self.batch_size = batch_size

        self.stage = stage
        self.num_gpus = num_gpus
        self.target_size = len(self.id_to_tag)
        self.use_crf = use_crf

        # set the default baseline model here
        self.pad_token_id = pad_token_id

        self.encoder_model = encoder_model
        self.encoder = AutoModelForTokenClassification.from_pretrained(encoder_model, num_labels=self.target_size, classifier_dropout=dropout_rate)
        if self.use_crf:
            self.crf_layer = ConditionalRandomField(num_tags=self.target_size, constraints=allowed_transitions(constraint_type="BIO", labels=self.id_to_tag))
        
        self.auxiliary_classifier = nn.Linear(self.encoder.config.hidden_size, 2)


        self.lr = lr
        self.span_f1 = SpanF1()
        self.val_span_f1 = SpanF1()
        self.setup_model(self.stage)
        self.save_hyperparameters('pad_token_id', 'encoder_model', 'use_crf')

    def setup_model(self, stage_name):
        if stage_name == 'fit' and self.train_data is not None:
            # Calculate total steps
            train_batches = len(self.train_data) // (self.batch_size * self.num_gpus)
            self.total_steps = 50 * train_batches

            self.warmup_steps = int(self.total_steps * 0.01)

    
    def collate_batch(self, mode='val'):
        if mode == 'fit':
            return self.train_collate_batch
        else:
            return self.train_collate_batch

    def train_collate_batch(self, batch):
        batch_ = list(zip(*batch))
        tokens, masks, gold_spans, tags, subtoken_pos_to_raw_pos, token_type_ids = batch_[0], batch_[1], batch_[2], batch_[3], batch_[4], batch_[5]

        max_len = max([len(token) for token in tokens])
        token_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.pad_token_id)
        # -100 is the default ignore index
        tag_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(-100)
        mask_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)
        tag_len_tensor = torch.zeros(size=(len(tokens),), dtype=torch.long)
        token_type_ids_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(0)
        auxiliary_tag_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(-100)


        for i in range(len(tokens)):
            tokens_ = tokens[i]
            seq_len = len(tokens_)
            tag_len = len(tags[i])
            token_tensor[i, :seq_len] = tokens_
            tag_len_tensor[i] = tag_len
            
            tag_tensor[i, :tag_len] = tags[i]
            auxiliary_tag_tensor[i, :tag_len] = torch.tensor([0 if self.id_to_tag[j.item()] == 'O' else 1 for j in tags[i]])
            mask_tensor[i, :seq_len] = masks[i]
            token_type_ids_tensor[i, :seq_len] = token_type_ids[i]

        return token_tensor, tag_tensor, mask_tensor, token_type_ids_tensor, gold_spans, subtoken_pos_to_raw_pos, tag_len_tensor, auxiliary_tag_tensor

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        if self.stage == 'fit':
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_steps)
            scheduler = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [scheduler]
        return [optimizer]

    def train_dataloader(self):
        loader = DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_batch(mode='train'), num_workers=2, shuffle=True)
        return loader

    def val_dataloader(self):
        if self.dev_data is None:
            return None
        loader = DataLoader(self.dev_data, batch_size=self.batch_size, collate_fn=self.collate_batch(mode='val'), num_workers=2)
        return loader

    def test_epoch_end(self, outputs):
        pred_results = self.span_f1.get_metric()
        avg_loss = np.mean([preds['loss'].item() for preds in outputs])
        self.log_metrics(pred_results, loss=avg_loss, on_step=False, on_epoch=True)

        out = {"test_loss": avg_loss, "results": pred_results}
        return out

    def training_epoch_end(self, outputs: List[Any]) -> None:
        pred_results = self.span_f1.get_metric(True)
        avg_loss = np.mean([preds['loss'].item() for preds in outputs])
        self.log_metrics(pred_results, loss=avg_loss, suffix='', on_step=False, on_epoch=True)
        self.span_f1.reset()

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        pred_results = self.val_span_f1.get_metric(True)
        avg_loss = np.mean([preds['loss'].item() for preds in outputs])
        self.log_metrics(pred_results, loss=avg_loss, suffix='val_', on_step=False, on_epoch=True)
        self.val_span_f1.reset()

    def validation_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch, mode='val')
        self.log_metrics(output['results'], loss=output['loss'], suffix='val_', on_step=True, on_epoch=False)
        return output

    def training_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch)
        self.log_metrics(output['results'], loss=output['loss'], suffix='', on_step=True, on_epoch=False)
        return output

    def test_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch, mode=self.stage)
        self.log_metrics(output['results'], loss=output['loss'], suffix='_t', on_step=True, on_epoch=False)
        return output

    def log_metrics(self, pred_results, loss=0.0, suffix='', on_step=False, on_epoch=True):
        for key in pred_results:
            self.log(suffix + key, pred_results[key], on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

        self.log(suffix + 'loss', loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

    def perform_forward_step(self, batch, mode=''):
        tokens, tags, token_mask, token_type_ids, metadata, subtoken_pos_to_raw_pos, tag_len, auxiliary_tag = batch
        batch_size = tokens.size(0)

        outputs = self.encoder(input_ids=tokens, attention_mask=token_mask, labels=tags, output_hidden_states=True)
        hidden_states = outputs.hidden_states[0]
        auxiliary_logits = self.auxiliary_classifier(hidden_states)
        loss_fct = CrossEntropyLoss()
        auxiliary_loss = loss_fct(auxiliary_logits.view(-1, 2), auxiliary_tag.view(-1))

        # compute the log-likelihood loss and compute the best NER annotation sequence
        token_scores = outputs.logits
        loss = 0.7 * outputs.loss + 0.3 * auxiliary_loss
        output = self._compute_token_tags(token_scores=token_scores, tags=tags, token_mask=token_mask, 
                                          metadata=metadata, subtoken_pos_to_raw_pos=subtoken_pos_to_raw_pos, batch_size=batch_size, mode=mode, tag_lens=tag_len)
        if not output['loss']:
            output['loss'] = loss
        return output

    def _compute_token_tags(self, token_scores, tags, token_mask, metadata, subtoken_pos_to_raw_pos, batch_size, tag_lens, mode=''):
        if self.use_crf:
        # compute the log-likelihood loss and compute the best NER annotation sequence
            loss = -self.crf_layer(token_scores, tags, token_mask) / float(batch_size)
            best_path = self.crf_layer.viterbi_tags(token_scores, token_mask)
        else:
            loss = None
            best_path = torch.argmax(token_scores, -1)

        pred_results = []
        raw_pred_results = []
        for i in range(batch_size):
            tag_len = tag_lens[i].item()
            if self.use_crf:
                tag_seq, _ = best_path[i]
                tag_seq = tag_seq[:tag_len]
            else:
                tag_seq = best_path[i].cpu().numpy()[0:tag_len]
            pred_results.append(extract_spans([self.id_to_tag[x] for x in tag_seq if x in self.id_to_tag], subtoken_pos_to_raw_pos))
            raw_pred_results.append([self.id_to_tag[x] for x in tag_seq if x in self.id_to_tag])
        output = {"loss": loss, "pred_results": pred_results, "raw_pred_results": raw_pred_results}
        if mode == 'val':
            self.val_span_f1(pred_results, metadata)
            output["results"] = self.val_span_f1.get_metric()
        else:
            self.span_f1(pred_results, metadata)
            output["results"] = self.span_f1.get_metric()

        return output

    def predict_tags(self, batch, tokenizer=None):
        tokens, tags, token_mask, metadata = batch
        pred_tags = self.perform_forward_step(batch, mode='predict')['token_tags']
        token_results, tag_results = [], []
        for i in range(tokens.size(0)):
            instance_token_results, instance_tag_results = get_tags(tokens[i], pred_tags[i], tokenizer=tokenizer)
            token_results.append(instance_token_results)
            tag_results.append(instance_tag_results)
        return token_results, tag_results


if __name__ == "__main__":
    from utils.util import get_reader, train_model, create_model, save_model, parse_args, get_tagset, wnut_iob, write_submit_result, load_model, get_entity_vocab
    import time
    import os
    base_dir = ""
    encoder_model = "distilbert-base-uncased"
    encoder_model = "bert-base-uncased"
    track = "EN-English/en"
    train_file = os.path.join(base_dir, "training_data/{}_train.conll".format(track))
    dev_file = os.path.join(base_dir, "training_data/{}_dev.conll".format(track))
    output_dir = os.path.join(base_dir, "{}".format(track), "{}-train".format(encoder_model))
    submission_file = os.path.join(base_dir, "submission", "{}.pred.conll".format(track))
    iob_tagging = wnut_iob
    entity_vocab = get_entity_vocab()
    train_data = get_reader(file_path=dev_file, target_vocab=iob_tagging, encoder_model=encoder_model, max_instances=-1, max_length=100, entity_vocab=entity_vocab, augment=[])
    entity_vocab = get_entity_vocab(conll_files=[train_file])
    dev_data = get_reader(file_path=dev_file, target_vocab=wnut_iob, encoder_model=encoder_model, max_instances=-1, max_length=55, augment=[])

    model = create_model(train_data=train_data, dev_data=dev_data, tag_to_id=train_data.get_target_vocab(),
                     dropout_rate=0.1, batch_size=16, stage='fit', lr=2e-5,
                     encoder_model=encoder_model, num_gpus=1, use_crf=False)

    trainer = train_model(model=model, out_dir=output_dir, epochs=20, monitor="f1")
    # use pytorch lightnings saver here.
    out_model_path, best_checkpoint = save_model(trainer=trainer, out_dir=output_dir, model_name=encoder_model, timestamp=time.time())

    model = load_model(best_checkpoint, wnut_iob, use_crf=False)

    record_data = write_submit_result(model, dev_data, submission_file)
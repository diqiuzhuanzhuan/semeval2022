from collections import defaultdict
import collections
from re import U
from typing import Dict, List, Tuple
from numpy.core.fromnumeric import size
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from transformers import AutoTokenizer
from transformers.models.luke.tokenization_luke import LukeTokenizer
from transformers.pipelines import token_classification
from transformers import LukeForEntitySpanClassification
from log import logger
from utils.reader_utils import get_ner_reader, extract_spans, _assign_ner_tags

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LukeCoNLLReader(Dataset):
    def __init__(self, max_instances=32, max_length=-1, target_vocab=None, pretrained_dir='', encoder_model='xlm-roberta-large'):
        self._max_instances = max_instances
        self._max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir + encoder_model, task="entity_span_classification")

        self.cls_token = self.tokenizer.special_tokens_map['cls_token']
        self.cls_token_id = self.tokenizer.get_vocab()[self.cls_token]
        self.pad_token = self.tokenizer.special_tokens_map['pad_token']
        self.pad_token_id = self.tokenizer.get_vocab()[self.pad_token]
        self.sep_token = self.tokenizer.special_tokens_map['sep_token']
        self.sep_token_id = self.tokenizer.get_vocab()[self.sep_token]

        self.label_to_id = {} if target_vocab is None else target_vocab
        self.instances = []
        # split into many pieces, ease --> e ase
        self.word_piece_ids = []
        self.pos_to_single_word_maps = []
        self.ner_tags = []
        self.type_count = defaultdict(int)
        self.max_len_entity = 0

    def get_target_size(self):
        return len(set(self.label_to_id.values()))

    def get_target_vocab(self):
        return self.label_to_id

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        fields = self.instances[item]
        sentence_str, entity_spans, labels, tokens, ner_tags = self.parse_line_for_ner(fields=fields)
        return sentence_str, entity_spans, labels, tokens, ner_tags

    def read_data(self, data):
        dataset_name = data if isinstance(data, str) else 'dataframe'
        logger.info('Reading file {}'.format(dataset_name))
        instance_idx = 0

        for fields, metadata in get_ner_reader(data=data):
            if self._max_instances != -1 and instance_idx > self._max_instances:
                break
            self.instances.append(fields)
            instance_idx += 1
        logger.info('Finished reading {:d} instances from file {}'.format(len(self.instances), dataset_name))

    def parse_line_for_ner(self, fields):
        tokens_, ner_tags = fields[0], fields[-1]
        sentence_str, entity_spans, labels = self.parse_tokens_for_ner(tokens_, ner_tags)

        return sentence_str, entity_spans, labels, tokens_, ner_tags

    def _get_entity_spans(self, sentence: str, word_positions: List[Tuple], entity_span_labels: Dict):
        start_positions = [_[0] for _ in word_positions]
        end_positions = [_[1] for _ in word_positions]
        entity_spans = []
        labels = []
        for i, start_pos in enumerate(start_positions):
            for end_pos in end_positions[i:]:
                if sentence[start_pos: end_pos].count(" ") > 7:
                    continue
                entity_spans.append((start_pos, end_pos))
                labels.append(self.label_to_id[entity_span_labels.get((start_pos, end_pos), 'O')])
        return entity_spans, labels

    def parse_tokens_for_ner(self, tokens_, ner_tags):
        sentence_str = ''
        entity_spans_labels = dict()
        word_positions = [] # for luke: he is happy---->[(0,2), (3, 5),(6, 11)]
        entity_start = None
        entity_label = None
        entity_end = None
        for idx, token in enumerate(tokens_):
            if sentence_str:
                sentence_str += " " + token.lower()
            else:
                sentence_str = token.lower()
            _word_position = (len(sentence_str) - len(token), len(sentence_str))
            word_positions.append(_word_position)
            assert(sentence_str[_word_position[0]: _word_position[1]] == token.lower())

            # if we have a NER here, in the case of B, the first NER tag is the B tag, the rest are I tags.
            ner_tag = ner_tags[idx]
            if ner_tag.startswith("B-"):
                if entity_label:
                    entity_end = _word_position[0] - 1
                    entity_spans_labels[(entity_start, entity_end)] = entity_label
                entity_start = _word_position[0]
                entity_label = ner_tag[2:]
                self.type_count[ner_tag[2:]] += 1
            if ner_tag.startswith("I-"):
                pass
            if ner_tag.startswith("O"):
                if entity_label:
                    entity_end = _word_position[0] - 1
                    entity_spans_labels[(entity_start, entity_end)] = entity_label
                entity_label = None
        if entity_label:
            entity_end = _word_position[1]
            entity_spans_labels[(entity_start, entity_end)] = entity_label
            entity_label = None
        # debug

        entity_spans, labels = self._get_entity_spans(sentence_str, word_positions, entity_spans_labels)
        self.max_len_entity = max(self.max_len_entity, len(entity_spans))

        return sentence_str, entity_spans, labels
    


if __name__ == "__main__":
    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base", task="entity_span_classification")
    uncased_entity_vocab = dict()
    for k in tokenizer.entity_vocab:
        uncased_entity_vocab[str.lower(k)] = tokenizer.entity_vocab[k]
    for k in uncased_entity_vocab:
        tokenizer.entity_vocab[k] = uncased_entity_vocab[k]
    def collate_batch(batch):
        batch_ = list(zip(*batch))
        sentence_str, entity_spans, labels, tokens, ner_tags = batch_[0], batch_[1], batch_[2], batch_[3], batch_[4]
        batch_size = len(sentence_str)
        max_label_len = max([len(token) for token in labels])
        labels_tensor = torch.empty(size=(batch_size, max_label_len), dtype=torch.long).fill_(luke_iob['O'])
        for i in range(batch_size):
            length = len(labels[i])
            labels_tensor[i, :length] = torch.tensor(labels[i])
        outputs = tokenizer(sentence_str, entity_spans=entity_spans, return_tensors="pt", max_length=100, padding=True, max_entity_length=max_label_len)
        outputs["labels"] = labels_tensor
        return outputs, tokens, ner_tags, entity_spans, sentence_str

    from utils.util import luke_iob
    conll_reader = LukeCoNLLReader(encoder_model="studio-ousia/luke-base", target_vocab=luke_iob)
    train_file = "./training_data/EN-English/en_train.conll"
    conll_reader.read_data(train_file)
    print(conll_reader.max_len_entity)
    data_loader = DataLoader(conll_reader, batch_size=1, collate_fn=collate_batch)
    model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-base", num_labels=len(luke_iob))
    from intervaltree import Interval, IntervalTree
    id_to_label = {v: k for k, v in luke_iob.items()}
    from utils.metric import SpanF1
    span_f1 = SpanF1()

    for batch in data_loader:
        inputs, tokens, ner_tags, entity_spans, sentence_str = batch
        outputs = model(**inputs)
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
                predict_res.append((span, id_to_label[predict_id]))
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

            assert(sentence == " ".join([_[0] for _ in final_res]))
            assert(len(ner_tags[i]) == len(final_res))
            final_label = [_[1] for _ in final_res] 
            y_true.append({(i, i+1): tag for i, tag in enumerate(ner_tags[i])})
            y_pred.append({(i, i+1): tag[1] for i, tag in enumerate(final_res)})
            batch_final_res.append(final_res)
        span_f1(y_true, y_pred)
        print(span_f1.get_metric())
        outputs["results"] = span_f1.get_metric()
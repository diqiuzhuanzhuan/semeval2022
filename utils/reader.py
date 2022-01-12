from collections import defaultdict
from pickle import LIST
from sys import prefix
from typing import Dict, List
import torch
from torch.utils.data import Dataset
import random

from transformers import AutoTokenizer, LukeTokenizer
import copy
import ahocorasick
from intervaltree import IntervalTree, Interval

from log import logger
from utils.reader_utils import get_ner_reader, extract_spans, _assign_ner_tags

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CoNLLReader(Dataset):
    def __init__(self, max_instances=-1, max_length=50, target_vocab=None, pretrained_dir='', encoder_model='xlm-roberta-large', entity_vocab: dict = None):
        self._max_instances = max_instances
        self._max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir + encoder_model)
        self.entity_vocab = entity_vocab
        if self.entity_vocab:
            self._setup_entity_vocab()

        self.cls_token = self.tokenizer.special_tokens_map['cls_token']
        self.cls_token_id = self.tokenizer.get_vocab()[self.cls_token]
        self.pad_token = self.tokenizer.special_tokens_map['pad_token']
        self.pad_token_id = self.tokenizer.get_vocab()[self.pad_token]
        self.sep_token = self.tokenizer.special_tokens_map['sep_token']
        self.sep_token_id = self.tokenizer.get_vocab()[self.sep_token]

        self.label_to_id = {} if target_vocab is None else target_vocab
        self.instances = []
        self.sentences = []
        # split into many pieces, ease --> e ase
        self.word_piece_ids = []
        self.pos_to_single_word_maps = []
        self.ner_tags = []
        self.type_count = defaultdict(int)
        self.type_to_entityset = defaultdict(set)
        self.label_entities = defaultdict(int)
    
    def _setup_entity_vocab(self):
        self.entity_automation = ahocorasick.Automaton()
        tmp = dict()
        for k in self.entity_vocab:
            self.entity_automation.add_word(k.lower(), (self.entity_vocab[k], k.lower()))
            tmp[k.lower()] = self.entity_vocab[k]
        for k in tmp:
            self.entity_vocab[k] = tmp[k]
        self.entity_automation.make_automaton()

    def _search_entity(self, sentence: str):
        ans = []
        words = set(sentence.split(" "))
        entity_pos = []
        tree = IntervalTree()
        for end_index, (insert_order, original_value) in self.entity_automation.iter(sentence):
            start_index = end_index - len(original_value) + 1
            if start_index > 1 and sentence[start_index-1] != " ":
                continue
            if end_index < len(sentence) - 1 and sentence[end_index+1] != " ":
                continue
            tree.remove_envelop(start_index, end_index)
            should_continue = False
            for item in tree.items():
                if start_index >= item.begin and end_index <= item.end:
                    should_continue = True
                    continue
            if should_continue:
                continue
            if original_value.count(" ") > 0:
                tree.add(Interval(start_index, end_index)) 
            elif original_value in words:
                if len(original_value) > 1:
                    tree.add(Interval(start_index, end_index)) 
        for interval in sorted(tree.items()):
            ans.append(sentence[interval.begin: interval.end+1])
            ans.append("$")
            entity_pos.append((interval.begin, interval.end))
        if len(ans) and ans[-1] == "$": 
            ans.pop(-1)
        return ans, entity_pos


    def get_target_size(self):
        return len(set(self.label_to_id.values()))

    def get_target_vocab(self):
        return self.label_to_id

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]
    
    def _wrap_data(self, fields):
        sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_, subtoken_pos_to_raw_pos, token_type_ids, prefix_location = self.parse_line_for_ner(fields=fields)
        self.sentences.append(sentence_str)
        tokens_tensor = torch.tensor(tokens_sub_rep, dtype=torch.long)
        #tag_tensor = torch.tensor(coded_ner_, dtype=torch.long).unsqueeze(0)
        tag_tensor = torch.tensor(coded_ner_, dtype=torch.long)
        token_masks_rep = torch.tensor(token_masks_rep)
        token_type_ids_tensor = torch.tensor(token_type_ids, dtype=torch.long)
        return tokens_tensor, token_masks_rep, gold_spans_, tag_tensor, subtoken_pos_to_raw_pos, token_type_ids_tensor

    def read_data(self, data):
        dataset_name = data if isinstance(data, str) else 'dataframe'
        logger.info('Reading file {}'.format(dataset_name))
        instance_idx = 0

        for fields, metadata in get_ner_reader(data=data):
            if self._max_instances != -1 and instance_idx > self._max_instances:
                break
            tokens_tensor, token_masks_rep, gold_spans_, tag_tensor, subtoken_pos_to_raw_pos, token_type_ids_tensor = self._wrap_data(fields)
            self.instances.append((tokens_tensor, token_masks_rep, gold_spans_, tag_tensor, subtoken_pos_to_raw_pos, token_type_ids_tensor))
            instance_idx += 1
        logger.info('Finished reading {:d} instances from file {}'.format(len(self.instances), dataset_name))

    def augment_data(self, data, type2type: dict):
        if len(self.type_to_entityset) == 0:
            logger.warning('Please run read_data before running augment_data')
            return
        dataset_name = data if isinstance(data, str) else 'dataframe'
        logger.info('Reading file {} for data augmentation'.format(dataset_name))
        for fields, metadata in get_ner_reader(data=data):
            tokens_, ner_tags = fields[0], fields[-1]
            new_tokens_, new_ner_tags = [], []
            for token, ner_tag in zip(tokens_, ner_tags):
                tag = ner_tag[2:] or "O"
                if tag not in type2type:
                    new_tokens_.append(token)
                    new_ner_tags.append(ner_tag)
                    continue
                rep_tag = type2type[tag]
                if ner_tag.startswith("B-"):
                    entity = random.sample(self.type_to_entityset[rep_tag], 1)[0]
                    new_tokens_.extend(entity.split(" "))
                    for i in range(len(entity.split(" "))):
                        if i == 0:
                            new_ner_tags.append("B-{}".format(tag))
                        else:
                            new_ner_tags.append("I-{}".format(tag))
            tokens_tensor, token_masks_rep, gold_spans_, tag_tensor, subtoken_pos_to_raw_pos, token_type_ids_tensor = self._wrap_data((new_tokens_, new_ner_tags))
            self.instances.append((tokens_tensor, token_masks_rep, gold_spans_, tag_tensor, subtoken_pos_to_raw_pos, token_type_ids_tensor))
        logger.info('Finished reading {:d} instances from file {}'.format(len(self.instances), dataset_name))

    def _entity_record(self, fields):
        tokens_, ner_tags = fields[0], fields[-1]
        entity = ""
        tag = ""
        for token, ner_tag in zip(tokens_, ner_tags):
            if ner_tag.startswith("B-"):
                entity = token
                tag = ner_tag[2:]
            elif ner_tag.startswith("I-") or ner_tag.startswith("E-"):
                entity = entity + " " + token
                assert(tag == ner_tag[2:])
            elif ner_tag.startswith("O"):
                if entity:
                    self.label_entities[entity] += 1
                    self.type_to_entityset[tag].add(entity)
                    entity = ""  
            elif ner_tag.startswith("S-"):
                entity = token
                tag = ner_tag[2:]
                self.label_entities[entity] += 1
                self.type_to_entityset[tag].add(entity)
                entity = ""
        if entity:
            self.label_entities[entity] += 1
            self.type_to_entityset[tag].add(entity)

    def parse_line_for_ner(self, fields):
        self._entity_record(fields)
        tokens_, ner_tags = fields[0], fields[-1]
        sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep, subtoken_pos_to_raw_pos, token_type_ids, prefix_location = self.parse_tokens_for_ner(tokens_, ner_tags)
        gold_spans_ = extract_spans(ner_tags_rep, subtoken_pos_to_raw_pos)
        coded_ner_ = [self.label_to_id[tag] for tag in ner_tags_rep]

        return sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_, subtoken_pos_to_raw_pos, token_type_ids, prefix_location

    def _add_prefix(self, tokens_, ner_tags, entity_pos):
        new_tokens_ = []
        new_ner_tags = []
        prefix_location = []
        sentence_str = " ".join(tokens_)
        prefix_insert_set = set()
        for begin, end in entity_pos:
            word_begin_index = sentence_str[0:begin].count(" ")
            prefix_insert_set.add(word_begin_index)
            word_end_index = sentence_str[0:end+1].count(" ")
            prefix_insert_set.add(word_end_index+1)
            
        for idx, token in enumerate(tokens_):
            if idx in prefix_insert_set:
                new_tokens_.append("$")
                new_ner_tags.append("O")
                prefix_location.append(len(new_tokens_)-1)
            new_tokens_.append(token)
            new_ner_tags.append(ner_tags[idx])
        if idx+1 in prefix_insert_set:
            new_tokens_.append("$")
            new_ner_tags.append("O")
            prefix_location.append(len(new_tokens_)-1)
        return new_tokens_, new_ner_tags, prefix_location

    def parse_tokens_for_ner(self, tokens_, ner_tags):
        if self.entity_vocab:
            sentence_str = " ".join(tokens_)
            _, entity_pos = self._search_entity(sentence_str)
            tokens_, ner_tags, prefix_location = self._add_prefix(tokens_, ner_tags, entity_pos)
        sentence_str = ''
        tokens_sub_rep, ner_tags_rep = [self.cls_token_id], ['O']
        token_type_ids = []
        pos_to_single_word = dict()
        subtoken_pos_to_raw_pos = []
        subtoken_pos_to_raw_pos.append(0)
        for idx, token in enumerate(tokens_):
            if self._max_length != -1 and len(tokens_sub_rep) > self._max_length:
                break
                
            if sentence_str:
                sentence_str += " " + token.lower()
            else:
                sentence_str = token.lower()
            if idx == 0:
                rep_ = self.tokenizer(token.lower())['input_ids']
            else:
                rep_ = self.tokenizer(" " + token.lower())['input_ids']
            rep_ = rep_[1:-1] #why? the first id is <s>, and the last id is </s>, so we eliminate them
            pos_to_single_word[(len(tokens_sub_rep), len(tokens_sub_rep)+len(rep_))] = token
            subtoken_pos_to_raw_pos.extend([idx+1] * len(rep_))
            tokens_sub_rep.extend(rep_)

            # if we have a NER here, in the case of B, the first NER tag is the B tag, the rest are I tags.
            ner_tag = ner_tags[idx]
            if ner_tag.startswith("B-"):
                self.type_count[ner_tag[2:]] += 1
            tags, masks = _assign_ner_tags(ner_tag, rep_)
            ner_tags_rep.extend(tags)
        self.pos_to_single_word_maps.append(pos_to_single_word)
        tokens_sub_rep.append(self.sep_token_id)
        token_type_ids.extend([0] * len(tokens_sub_rep))
        subtoken_pos_to_raw_pos.append(idx+2)
        assert(self.tokenizer(sentence_str)["input_ids"] == tokens_sub_rep)
        ner_tags_rep.append('O')
        self.ner_tags.append(ner_tags_rep)
        if self.entity_vocab:
            entity_ans, _ = self._search_entity(sentence_str)
            for idx, token in enumerate(entity_ans):
                if self._max_length != -1 and len(tokens_sub_rep) > self._max_length:
                    break
                if idx == 0:
                    rep_ = self.tokenizer(token.lower())['input_ids']
                else:
                    rep_ = self.tokenizer(" " + token.lower())['input_ids']
                rep_ = rep_[1:-1] #why? the first id is <s>, and the last id is </s>, so we eliminate them
                tokens_sub_rep.extend(rep_)
            
            tokens_sub_rep.append(self.sep_token_id)
        token_masks_rep = [True] * len(tokens_sub_rep)
        token_type_ids.extend([1] * (len(tokens_sub_rep) - len(token_type_ids)))
        #assert(token_masks_rep == self.tokenizer(sentence_str)["attention_mask"])
        
        return sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep, subtoken_pos_to_raw_pos, token_type_ids, prefix_location


if __name__ == "__main__":
    from utils.util import wnut_iob
    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
    entity_vocab = copy.deepcopy(tokenizer.entity_vocab)
    conll_reader = CoNLLReader(encoder_model="roberta-base", target_vocab=wnut_iob, entity_vocab=entity_vocab)
    train_file = "./training_data/EN-English/en_train.conll"
    dev_file = "./training_data/EN-English/en_dev.conll"
    conll_reader.read_data(train_file)
    #conll_reader.augment_data(train_file, {"CORP": "GRP"})
    #conll_reader.augment_data(train_file, {"GRP": "CORP"})
    for batch in conll_reader:
        pass
        #print(batch)
        

            
    
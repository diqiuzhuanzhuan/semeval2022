import argparse
import collections
from json import encoder
import os
import time
from typing import Union, List
import pandas as pd
import torch
from pytorch_lightning import seed_everything

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.cuda.memory import reset_accumulated_memory_stats
import transformers
from torch.utils.data import DataLoader
from log import logger
from model.ner_model import NERBaseAnnotator
from model.luke_model import LukeNer, negative_sample
from utils.reader import CoNLLReader
from utils.conll_reader import LukeCoNLLReader
from utils.reader_utils import get_ner_reader

conll_iob = {'B-ORG': 0, 'I-ORG': 1, 'B-MISC': 2, 'I-MISC': 3, 'B-LOC': 4, 'I-LOC': 5, 'B-PER': 6, 'I-PER': 7, 'O': 8}
wnut_iob = {'B-CORP': 0, 'I-CORP': 1, 'B-CW': 2, 'I-CW': 3, 'B-GRP': 4, 'I-GRP': 5, 'B-LOC': 6, 'I-LOC': 7, 'B-PER': 8, 'I-PER': 9, 'B-PROD': 10, 'I-PROD': 11, 'O': 12}
luke_iob = {'CORP': 0, 'CW': 1, 'GRP': 2, 'LOC':3, 'PER': 4, 'PROD': 5, 'O': 6}


def parse_args():
    p = argparse.ArgumentParser(description='Model configuration.', add_help=False)
    p.add_argument('--train', type=str, help='Path to the train data.', default=None)
    p.add_argument('--test', type=str, help='Path to the test data.', default=None)
    p.add_argument('--dev', type=str, help='Path to the dev data.', default=None)

    p.add_argument('--out_dir', type=str, help='Output directory.', default='.')
    p.add_argument('--iob_tagging', type=str, help='IOB tagging scheme', default='wnut')

    p.add_argument('--max_instances', type=int, help='Maximum number of instances', default=-1)
    p.add_argument('--max_length', type=int, help='Maximum number of tokens per instance.', default=50)

    p.add_argument('--encoder_model', type=str, help='Pretrained encoder model to use', default='xlm-roberta-large')
    p.add_argument('--model', type=str, help='Model path.', default=None)
    p.add_argument('--model_name', type=str, help='Model name.', default=None)
    p.add_argument('--stage', type=str, help='Training stage', default='fit')
    p.add_argument('--prefix', type=str, help='Prefix for storing evaluation files.', default='test')

    p.add_argument('--batch_size', type=int, help='Batch size.', default=128)
    p.add_argument('--gpus', type=int, help='Number of GPUs.', default=1)
    p.add_argument('--epochs', type=int, help='Number of epochs for training.', default=5)
    p.add_argument('--lr', type=float, help='Learning rate', default=1e-5)
    p.add_argument('--dropout', type=float, help='Dropout rate', default=0.1)

    return p.parse_args()


def get_tagset(tagging_scheme):
    if 'conll' in tagging_scheme:
        return conll_iob
    return wnut_iob


def get_out_filename(out_dir, model, prefix):
    model_name = os.path.basename(model)
    model_name = model_name[:model_name.rfind('.')]
    return '{}/{}_base_{}.tsv'.format(out_dir, prefix, model_name), '{}/{}_base_{}_detail.csv'.format(out_dir, prefix, model_name)


def write_eval_performance(eval_performance, out_file):
    outstr = ''
    added_keys = set()
    for out_ in eval_performance:
        for k in out_:
            if k in added_keys or k in ['results', 'predictions']:
                continue
            outstr = outstr + '{}\t{}\n'.format(k, out_[k])
            added_keys.add(k)

    open(out_file, 'wt').write(outstr)
    logger.info('Finished writing evaluation performance for {}'.format(out_file))

def write_submit_result(model: Union[NERBaseAnnotator, LukeNer], test_data: Union[CoNLLReader, LukeCoNLLReader], out_file: str):
    path = os.path.dirname(out_file)
    if path and not os.path.exists(path):
        os.makedirs(path)
    batch_size = 8
    test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=model.collate_batch(mode='val'))
    sentences = test_data.sentences
    ner_tags = test_data.ner_tags
    pos_to_singel_word_map = test_data.pos_to_single_word_maps
    f = open(out_file, "w")
    #
    record_data = collections.defaultdict(list)
    for idx, batch in enumerate(test_dataloader):
        output = model.perform_forward_step(batch)
        pred_result = output["pred_results"]
        if isinstance(model, LukeNer):
            ner_tags_ = batch[2]
            for i in range(len(pred_result)):
                for (word, pred_label), y_true in zip(pred_result[i], ner_tags_[i]):
                    record_data["word"].append(word)
                    record_data["label"].append(y_true)
                    record_data["pred"].append(pred_label)
                    f.write("{}\n".format(pred_label))
                f.write("\n")
            
            continue
        raw_pred_results = output["raw_pred_results"]
        for i in range(len(pred_result)):
            sentence = sentences[idx*batch_size+i]
            pos_to_singel_word = pos_to_singel_word_map[idx*batch_size+i]
            ner_tag = ner_tags[idx*batch_size+i]
            input_ids = batch[0][i]
            pred_token_tag = pred_result[i]
            raw_pred_token_tag = raw_pred_results[i]
            metadata_token_tag = batch[4][i]
            meta_labels = []
            pred_labels = []
            sentence_subtokens = test_data.tokenizer.convert_ids_to_tokens(input_ids)
            for (start_pos, end_pos), (pred_start_pos, pred_end_pos) in zip(metadata_token_tag, pred_token_tag):
                sub_tokens = test_data.tokenizer.convert_ids_to_tokens(input_ids[start_pos: end_pos+1])
                pred_sub_tokens = test_data.tokenizer.convert_ids_to_tokens(input_ids[start_pos: end_pos+1])
                tag = metadata_token_tag[(start_pos, end_pos)]
                pred_tag = pred_token_tag[(pred_start_pos, pred_end_pos)]
                for sub_token1, sub_token2 in zip(sub_tokens, pred_sub_tokens):
                    meta_labels.append(tag)
                    pred_labels.append(pred_tag)
                #f.write("{}{}{}{}{}{}{}".format(sub_token1, ",", tag, ",", sub_token2, ",", pred_tag))
                #f.write("\n")
            for (start, end) in pos_to_singel_word:
                single_word_tokens = sentence_subtokens[start:end]
                word = "".join(single_word_tokens).replace("##", "")
                word_meta_tag = ner_tag[start]
                word_pred_tag = raw_pred_token_tag[start]
                record_data["word"].append(word)
                record_data["label"].append(word_meta_tag)
                record_data["pred"].append(word_pred_tag)
                f.write("{}\n".format(word_pred_tag))
            f.write("\n")
    f.close()
    return pd.DataFrame(record_data)

def get_entity_vocab(encoder_model="studio-ousia/luke-base", conll_files: List[str]=[], entity_files: List[str]=[]):
    from transformers import LukeTokenizer
    import copy
    tokenizer = LukeTokenizer.from_pretrained(encoder_model)
    entity_vocab = copy.deepcopy(tokenizer.entity_vocab)
    def _get_entity(fields, entity_set):
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
                    entity_set.add(entity)
                    entity = ""  
            elif ner_tag.startswith("S-"):
                entity = token
                tag = ner_tag[2:]
                entity_set.add(entity)
                entity = ""
        if entity:
            entity_set.add(entity)

    for file in conll_files:
        entity_set = set()
        for fields, _ in get_ner_reader(file):
            _get_entity(fields, entity_set)
        
        for entity in entity_set:
            if entity not in entity_vocab:
                entity_vocab[entity] = len(entity_vocab)

    for file in entity_files:
        with open(file, "r") as f:
            for entity in f:
                entity = entity.strip('\r').strip('\n')
                if not entity:
                    continue
                if entity not in entity_vocab:
                    entity_vocab[entity] = len(entity_vocab)
    
    return entity_vocab


def get_reader(file_path, max_instances=-1, max_length=50, target_vocab=None, encoder_model='xlm-roberta-large', entity_vocab=None, augment=[{'GRP': 'CORP'}, {'CORP': 'GRP'}]):
    if file_path is None:
        return None
    if "luke" in encoder_model:
        reader = LukeCoNLLReader(max_instances=max_instances, max_length=max_length, target_vocab=target_vocab, encoder_model=encoder_model)
    else:
        reader = CoNLLReader(max_instances=max_instances, max_length=max_length, target_vocab=target_vocab, encoder_model=encoder_model, entity_vocab=entity_vocab)
    reader.read_data(file_path)
    for ele in augment:
        reader.augment_data(file_path, ele)

    return reader


def create_model(train_data, dev_data, tag_to_id, batch_size=64, dropout_rate=0.1, stage='fit', lr=1e-5, encoder_model='xlm-roberta-large', num_gpus=1, use_crf=False, negative_sample=False):
    if "luke" in encoder_model:
        return LukeNer(encoder_model=encoder_model, batch_size=batch_size, lr=lr, dropout_rate=dropout_rate, train_data=train_data, dev_data=dev_data, tag_to_id=tag_to_id, negative_sample=negative_sample)
    else:
        return NERBaseAnnotator(train_data=train_data, dev_data=dev_data, tag_to_id=tag_to_id, batch_size=batch_size, stage=stage, encoder_model=encoder_model,
                            dropout_rate=dropout_rate, lr=lr, pad_token_id=train_data.pad_token_id, num_gpus=num_gpus, use_crf=use_crf)


def load_model(model_file, tag_to_id=None, stage='test', use_crf=False):
    hparams_file = model_file[:model_file.rindex('checkpoints/')] + '/hparams.yaml'
    if "luke"  in model_file:
        print("loading luke_model")
        model = LukeNer.load_from_checkpoint(model_file, hparams_file=hparams_file, stage=stage, tag_to_id=tag_to_id)
        model.stage = stage
    else:
        model = NERBaseAnnotator.load_from_checkpoint(model_file, hparams_file=hparams_file, stage=stage, tag_to_id=tag_to_id, use_crf=use_crf)
        model.stage = stage
    return model


def save_model(trainer: pl.Trainer, out_dir, model_name='', timestamp=None):
    out_dir = out_dir + '/lightning_logs/version_' + str(trainer.logger.version) + '/checkpoints/'
    if timestamp is None:
        timestamp = time.time()
    os.makedirs(out_dir, exist_ok=True)

    outfile = out_dir + '/' + model_name + '_timestamp_' + str(timestamp) + '_final.ckpt'
    trainer.save_checkpoint(outfile, weights_only=True)

    logger.info('Stored model {}.'.format(outfile))
    best_checkpoint = None
    for file in os.listdir(out_dir):
        if file.startswith("epoch"):
            best_checkpoint = os.path.join(out_dir, file)
            break
    return outfile, best_checkpoint


def train_model(model, out_dir='', epochs=10, gpus=1, monitor='val_loss'):
    trainer = get_trainer(gpus=gpus, out_dir=out_dir, epochs=epochs, monitor=monitor)
    trainer.fit(model)
    return trainer


def get_trainer(gpus=4, is_test=False, out_dir=None, epochs=10, monitor='val_loss'):
    seed_everything(42)
    if is_test:
        return pl.Trainer(gpus=1) if torch.cuda.is_available() else pl.Trainer(val_check_interval=100)

    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=gpus, max_epochs=epochs, callbacks=[get_model_earlystopping_callback(monitor), get_model_best_checkpoint_callback(out_dir, monitor)],
                             default_root_dir=out_dir, checkpoint_callback=True)
        trainer.callbacks.append(get_lr_logger())
    else:
        trainer = pl.Trainer(max_epochs=epochs, default_root_dir=out_dir)

    return trainer


def get_lr_logger():
    lr_monitor = LearningRateMonitor(logging_interval='step')
    return lr_monitor


def get_model_earlystopping_callback(monitor='val_loss'):
    if monitor == 'f1':
        es_clb = EarlyStopping(
            monitor='val_micro@F1',
            min_delta=0.001,
            patience=5,
            verbose=True,
            mode='max'
        )
    else:
        es_clb = EarlyStopping(
            monitor=monitor,
            min_delta=0.001,
            patience=5,
            verbose=True,
            mode='min'
        )
    return es_clb

    
def get_model_best_checkpoint_callback(dirpath='checkpoints', monitor='val_loss'):
    if monitor == "f1":
        bc_clb = ModelCheckpoint(
            filename='{epoch}-{val_micro@F1:.3f}-{val_loss:.2f}',
            save_top_k=1,
            verbose=True,
            monitor="val_micro@F1",
            mode="max"
        )
    else:
        bc_clb = ModelCheckpoint(
            filename='{epoch}-{val_micro@F1:.3f}-{val_loss:.2f}',
            save_top_k=1,
            verbose=True,
            monitor=monitor,
            mode="min"
        )
    return  bc_clb
        

if __name__ == "__main__":
    train_file = "./training_data/EN-English/en_train.conll"
    reader = get_reader(train_file, target_vocab=wnut_iob, encoder_model='roberta-base')
    wiki_file = "./data/wiki_def/wiki_abstract.vocab"
    entity_vocab = get_entity_vocab(conll_files=[train_file], entity_files=[wiki_file])
    print(len(entity_vocab))
import gzip
import itertools


def get_ner_reader(data):
    fin = gzip.open(data, 'rt') if data.endswith('.gz') else open(data, 'rt')
    for is_divider, lines in itertools.groupby(fin, _is_divider):
        if is_divider:
            continue
        lines = [line.strip().replace('\u200d', '').replace('\u200c', '') for line in lines]

        metadata = lines[0].strip() if lines[0].strip().startswith('# id') else None
        fields = [line.split() for line in lines if not line.startswith('# id')]
        fields = [list(field) for field in zip(*fields)]


        yield fields, metadata


def _assign_ner_tags(ner_tag, rep_):
    ner_tags_rep = []
    token_masks = []

    sub_token_len = len(rep_)
    token_masks.extend([True] * sub_token_len)
    if ner_tag[0] == 'B':
        in_tag = 'I' + ner_tag[1:]

        ner_tags_rep.append(ner_tag)
        ner_tags_rep.extend([in_tag] * (sub_token_len - 1))
    else:
        ner_tags_rep.extend([ner_tag] * sub_token_len)
    return ner_tags_rep, token_masks


def extract_spans(tags, subtoken_pos_to_raw_pos):
    cur_tag = None
    cur_start = None
    gold_spans = {}

    def _save_span(_cur_tag, _cur_start, _cur_id, _gold_spans):
        if _cur_start is None:
            return _gold_spans
        _gold_spans[(_cur_start, _cur_id - 1)] = _cur_tag  # inclusive start & end, accord with conll-coref settings
        return _gold_spans

    # iterate over the tags
    for _id, (nt, pos) in enumerate(zip(tags, subtoken_pos_to_raw_pos)):
        indicator = nt[0]
        """
        if _id == 0 or pos != subtoken_pos_to_raw_pos[_id-1]:
            new_word = True
        else:
            new_word = False
        if cur_tag and new_word:
            _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_start = _id
            cur_tag = nt
        elif new_word:
            cur_tag = nt
            cur_start = _id
        else:
            pass
        """
        if indicator == 'B':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_start = _id
            cur_tag = nt[2:]
            pass
        elif indicator == 'I':
            # do nothing
            pass
        elif indicator == 'O':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_tag = 'O'
            cur_start = _id
            pass
    _save_span(cur_tag, cur_start, _id + 1, gold_spans)
    return gold_spans


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True

    first_token = line.split()[0]
    if first_token == "-DOCSTART-":# or line.startswith('# id'):  # pylint: disable=simplifiable-if-statement
        return True

    return False


def get_tags(tokens, tags, tokenizer=None, start_token_pattern='▁'):
    token_results, tag_results = [], []
    index = 0
    token_word = []
    tokens = tokenizer.convert_ids_to_tokens(tokens)
    for token, tag in zip(tokens, tags):
        if token == tokenizer.pad_token:
            # index += 1
            continue

        if index == 0:
            tag_results.append(tag)

        elif token.startswith(start_token_pattern) and token != '▁́':
            tag_results.append(tag)

            if tokenizer is not None:
                token_results.append(''.join(token_word).replace(start_token_pattern, ''))
            token_word.clear()

        token_word.append(token)

        index += 1
    token_results.append(''.join(token_word).replace(start_token_pattern, ''))

    return token_results, tag_results
import re
import json
import spacy
import msgpack
import unicodedata
import numpy as np
import argparse
import collections
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from DRQA.model import str2bool
import logging


def main():
    args, log = setup()
    
    train = flatten_json(args.trn_file, 'train')
    dev = flatten_json(args.dev_file, 'dev')
    log.info('json data flattened.')

    # Tokenize and Annotate.
    with Pool(args.threads, initializer=init) as p:
        annotate_ = partial(annotate, wv_cased=args.wv_cased)
        train = list(tqdm(p.imap(annotate_, train, chunksize=args.batch_size), total=len(train), desc='train'))
        dev = list(tqdm(p.imap(annotate_, dev, chunksize=args.batch_size), total=len(dev), desc='dev  '))
    
    train = list(map(index_answer, train))
     # Number of data points in the training set.
    initial_len = len(train)

    # Filter out inconsistent samples
    train = list(filter(lambda x: x[-1] is not None, train))
    log.info('drop {} inconsistent samples.'.format(initial_len - len(train)))
    log.info('tokens generated')



    # load vocabulary from word vector files (GLOVE)
    wv_vocab = set()
    with open(args.wv_file, encoding="UTF-8") as f:
        for line in f:
            token = normalize_text(line.rstrip().split(' ')[0])
            # Add the word
            wv_vocab.add(token)
    log.info('glove vocab loaded.')


    # build vocabulary
    full = train + dev
    # Build the vocabulary.
    vocab, counter = build_vocab([row[5] for row in full], [row[1] for row in full], wv_vocab, args.sort_all)
    total = sum(counter.values())
    matched = sum(counter[t] for t in vocab)
    log.info('vocab coverage {1}/{0} | OOV occurrence {2}/{3} ({4:.4f}%)'.format(
        len(counter), len(vocab), (total - matched), total, (total - matched) / total * 100))
    counter_tag = collections.Counter(w for row in full for w in row[3])
    vocab_tag = sorted(counter_tag, key=counter_tag.get, reverse=True)
    counter_ent = collections.Counter(w for row in full for w in row[4])
    vocab_ent = sorted(counter_ent, key=counter_ent.get, reverse=True)
    

    # Convert words to their ID's
    # Convert Tag to the corresponding ID's
    # Convert Named Entity Tag to the corresponding ID.
    w2id = {w: i for i, w in enumerate(vocab)}
    tag2id = {w: i for i, w in enumerate(vocab_tag)}
    ent2id = {w: i for i, w in enumerate(vocab_ent)}
    
    
    log.info('Vocabulary size: {}'.format(len(vocab)))
    log.info('Found {} POS tags.'.format(len(vocab_tag)))
    log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))
    
    # Convert words to id
    to_id_ = partial(to_id, w2id=w2id, tag2id=tag2id, ent2id=ent2id)
    train = list(map(to_id_, train))
    dev = list(map(to_id_, dev))
    
    log.info('converted to ids.')

    vocab_size = len(vocab)
    # Storing all the word embeddings in the vocabulary.
    embeddings = np.zeros((vocab_size, args.wv_dim))
    # Cound the number of times a particular word has occured overall
    embed_counts = np.zeros(vocab_size)
    # First two Indexes are reserved for PADDING and UNKNOWN tokens
    embed_counts[:2] = 1  # PADDING & UNK
    
    with open(args.wv_file) as f:
        for line in f:
            elems = line.rstrip().split(' ')
            token = normalize_text(elems[0]) # Token
            if token in w2id:
                word_id = w2id[token]
                embed_counts[word_id] += 1
                embeddings[word_id] += [float(v) for v in elems[1:]]
    embeddings /= embed_counts.reshape((-1, 1))
    log.info('got embedding matrix.')

    meta = {
        'vocab': vocab,
        'vocab_tag': vocab_tag,
        'vocab_ent': vocab_ent,
        'embedding': embeddings.tolist(),
        'wv_cased': args.wv_cased,
    }
    with open('SQuAD/meta.msgpack', 'wb') as f:
        msgpack.dump(meta, f)
    result = {
        'train': train,
        'dev': dev
    }
    # train: id, context_id, context_features, tag_id, ent_id,
    #        question_id, context, context_token_span, answer_start, answer_end
    # dev:   id, context_id, context_features, tag_id, ent_id,
    #        question_id, context, context_token_span, answer
    with open('SQuAD/data.msgpack', 'wb') as f:
        msgpack.dump(result, f)
    if args.sample_size:
        sample = {
            'train': train[:args.sample_size],
            'dev': dev[:args.sample_size]
        }
        with open('SQuAD/sample.msgpack', 'wb') as f:
            msgpack.dump(sample, f)
    log.info('saved to disk.')

def setup():
    parser = argparse.ArgumentParser(
        description='Preprocessing data files, about 10 minitues to run.'
    )
    parser.add_argument('--trn_file', default='SQuAD/train-v1.1.json',
                        help='path to train file.')
    parser.add_argument('--dev_file', default='SQuAD/dev-v1.1.json',
                        help='path to dev file.')
    parser.add_argument('--wv_file', default='glove/glove.840B.300d.txt',
                        help='path to word vector file.')
    parser.add_argument('--wv_dim', type=int, default=300,
                        help='word vector dimension.')
    parser.add_argument('--wv_cased', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='treat the words as cased or not.')
    parser.add_argument('--sort_all', action='store_true',
                        help='sort the vocabulary by frequencies of all words. '
                             'Otherwise consider question words first.')
    parser.add_argument('--sample_size', type=int, default=0,
                        help='size of sample data (for debugging).')
    parser.add_argument('--threads', type=int, default=min(multiprocessing.cpu_count(), 16),
                        help='number of threads for preprocessing.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for multiprocess tokenizing and tagging.')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                        datefmt='%m/%d/%Y %I:%M:%S')
    log = logging.getLogger(__name__)
    log.info(vars(args))
    log.info('start data preparing...')

    return args, log




def flatten_json(data_file, mode):
    with open(data_file) as f:
        data = json.load(f)['data']
    rows = []
    for article in data:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                id_, question, answers = qa['id'], qa['question'], qa['answers']
                if mode == 'train':
                    answer = answers[0]['text']  # in training data there's only one answer
                    answer_start = answers[0]['answer_start']
                    answer_end = answer_start + len(answer)
                    rows.append((id_, context, question, answer, answer_start, answer_end))
                else:  # mode == 'dev'
                    answers = [a['text'] for a in answers]
                    rows.append((id_, context, question, answers))
    return rows




def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def clean_spaces(text):
    # Removes unnecessary spaces in a string.
    text = re.sub(r'\s', ' ', text)
    return text

nlp = None


def init():
    # Initialize Spacy.
    global nlp
    nlp = spacy.load('en_core_web_md')


def annotate(row, wv_cased):
    global nlp
    id_, context, question = row[:3]

    q_doc = nlp(clean_spaces(question))
    c_doc = nlp(clean_spaces(context))
    
    # Call the normalize function on the words in the sentence.
    question_tokens = [normalize_text(w.text) for w in q_doc]
    context_tokens = [normalize_text(w.text) for w in c_doc]
    
    # Convert the question and paragraph tokens into lower case
    question_tokens_lower = [w.lower() for w in question_tokens]
    context_tokens_lower = [w.lower() for w in context_tokens]
    
    # 
    context_token_span = [(w.idx, w.idx + len(w.text)) for w in c_doc]
    # Parts of Speech Tag
    context_tags = [w.tag_ for w in c_doc]
    # Named Entity Tags
    context_ents = [w.ent_type_ for w in c_doc]
    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in q_doc}
    question_tokens_set = set(question_tokens)
    question_tokens_lower_set = set(question_tokens_lower)
    match_origin = [w in question_tokens_set for w in context_tokens]
    match_lower = [w in question_tokens_lower_set for w in context_tokens_lower]
    match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in c_doc]
    # term frequency in document
    counter_ = collections.Counter(context_tokens_lower)
    total = len(context_tokens_lower)
    context_tf = [counter_[w] / total for w in context_tokens_lower]
    context_features = list(zip(match_origin, match_lower, match_lemma, context_tf))
    
    if not wv_cased:
        context_tokens = context_tokens_lower
        question_tokens = question_tokens_lower
    
    return (id_, context_tokens, context_features, context_tags, context_ents,
            question_tokens, context, context_token_span) + row[3:]


def index_answer(row):
    token_span = row[-4]
    starts, ends = zip(*token_span)
    answer_start = row[-2]
    answer_end = row[-1]
    try:
        return row[:-3] + (starts.index(answer_start), ends.index(answer_end))
    except ValueError:
        return row[:-3] + (None, None)


def build_vocab(questions, contexts, wv_vocab, sort_all=False):
    """
    Build vocabulary sorted by global word frequency, or consider frequencies in questions first,
    which is controlled by `args.sort_all`.
    """
    if sort_all:
        counter = collections.Counter(w for doc in questions + contexts for w in doc)
        vocab = sorted([t for t in counter if t in wv_vocab], key=counter.get, reverse=True)
    else:
        counter_q = collections.Counter(w for doc in questions for w in doc)
        counter_c = collections.Counter(w for doc in contexts for w in doc)
        counter = counter_c + counter_q
        vocab = sorted([t for t in counter_q if t in wv_vocab], key=counter_q.get, reverse=True)
        vocab += sorted([t for t in counter_c.keys() - counter_q.keys() if t in wv_vocab],
                        key=counter.get, reverse=True)
    vocab.insert(0, "<PAD>")
    vocab.insert(1, "<UNK>")
    return vocab, counter


def to_id(row, w2id, tag2id, ent2id, unk_id=1):
    context_tokens = row[1] # Words in the paragraph
    context_features = row[2]   # The Exact match between question and paragraph
    context_tags = row[3]   # POS TAG
    context_ents = row[4]   # Named Entity Recognition TAG
    question_tokens = row[5] # The tokens belonging to the question
   
    # Converting the question token to the IDs
    question_ids = [w2id[w] if w in w2id else unk_id for w in question_tokens]
    context_ids = [w2id[w] if w in w2id else unk_id for w in context_tokens]
    tag_ids = [tag2id[w] for w in context_tags]
    ent_ids = [ent2id[w] for w in context_ents]
    return (row[0], context_ids, context_features, tag_ids, ent_ids, question_ids) + row[6:]


if __name__ == '__main__':
    main()

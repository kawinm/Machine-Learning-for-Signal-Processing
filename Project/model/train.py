import re
import os
import sys
import math
import random
import string
import logging
import argparse
import torch
from shutil import copyfile
from datetime import datetime
from collections import Counter
import msgpack
from DRQA.model import DocReaderModel
from DRQA.model import str2bool


def main():
    args, log = setup()
    log.info('[Program starts. Loading data...]')
    train, dev, dev_y, embedding, opt = load_data(vars(args))
    
    if args.save_dawn_logs:
        dawn_start = datetime.now()
        log.info('dawn_entry: epoch\tf1Score\thours')

    model = DocReaderModel(opt, embedding)
    epoch_0 = 1
    best_val_score = 0.0

    for epoch in range(epoch_0, epoch_0 + args.epochs):
        log.warning('Epoch {}'.format(epoch))
        # Get the batches of data to train.
        batches = BatchGen(train, batch_size=args.batch_size, gpu=args.cuda)
        # Starting time
        start = datetime.now()
        for i, batch in enumerate(batches):
            # Iterate through all the batches and train the data on the model
            model.update(batch)
            
            if i % args.log_per_updates == 0:
                log.info('> epoch [{0:2}] updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(
                    epoch, model.updates, model.train_loss.value,
                    str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
                # Print loss value every log_per_updates
             

        log.debug('\n')
        
        # evaluation loop
        batches = BatchGen(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
        predictions = []
        for i, batch in enumerate(batches):
            predictions.extend(model.predict(batch))
            log.debug('> evaluating [{}/{}]'.format(i, len(batches)))
        
        
        # Get the EM and F1 score
        em, f1 = score(predictions, dev_y)
        log.warning("dev EM: {} F1: {}".format(em, f1))
        
        if args.save_dawn_logs:
            time_diff = datetime.now() - dawn_start
            log.warning("dawn_entry: {}\t{}\t{}".format(epoch, f1/100.0, float(time_diff.total_seconds() / 3600.0)))
       
        # save
        if not args.save_last_only or epoch == epoch_0 + args.epochs - 1:
            
            # Get the new model
            model_file = os.path.join(args.model_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
            # Save the model.
            model.save(model_file, epoch, [em, f1, best_val_score])
            
            # Save the new model if the EM and F1 scores are better
            if f1 > best_val_score:
                best_val_score = f1
                copyfile(
                    model_file,
                    os.path.join(args.model_dir, 'best_model.pt'))
                log.info('[new best model saved.]')


def setup():
    parser = argparse.ArgumentParser(
        description='Train the Document Reader Model.'
    )
    # Arguments 
    parser.add_argument('--log_per_updates', type=int, default=100,
                        help='log model loss per x updates (mini-batches).')
    parser.add_argument('--data_file', default='SQuAD/data.msgpack',
                        help='path to preprocessed data file.')
    parser.add_argument('--model_dir', default='models',
                        help='path to store saved models.')
    parser.add_argument('--save_last_only', action='store_true',
                        help='only save the final models.')
    parser.add_argument('--save_dawn_logs', action='store_true',
                        help='append dawnbench log entries prefixed with dawn_entry:')
    parser.add_argument('--seed', type=int, default=1013,
                        help='random seed for data shuffling, dropout, etc.')
    parser.add_argument("--cuda", type=str2bool, nargs='?',
                        const=True, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    
    # Arguments for training.
    parser.add_argument('-e', '--epochs', type=int, default=40)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-rlr', '--reduce_lr', type=float, default=0.,
                        help='reduce initial (resumed) learning rate by this factor.')
    parser.add_argument('-op', '--optimizer', default='adam',
                        help='supported optimizer: adamax, sgd, adam')
    parser.add_argument('-gc', '--grad_clipping', type=float, default=10)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1,
                        help='only applied to SGD.')
    parser.add_argument('-mm', '--momentum', type=float, default=0,
                        help='only applied to SGD.')
    parser.add_argument('-tp', '--tune_partial', type=int, default=1000,
                        help='finetune top-x embeddings.')
    parser.add_argument('--fix_embeddings', action='store_true',
                        help='if true, `tune_partial` will be ignored.')
    parser.add_argument('--rnn_padding', action='store_true',
                        help='perform rnn padding (much slower but more accurate).')
    # model
    parser.add_argument('-model', default='coat_self',
                        help='Model Architecture types: base (Base DrQA model), coat (Base + Coattention), coat_self (Base + Coattention + Self Attention), triple_self_coat (Base + Coattention + ScaledDotProductAttention) ')
    #parser.add_argument('--question_merge', default='self_attn')
    parser.add_argument('--doc_layers', type=int, default=3)
    parser.add_argument('--question_layers', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_features', type=int, default=4)
    parser.add_argument('--pos', type=str2bool, nargs='?', const=True, default=True,
                        help='use pos tags as a feature.')
    parser.add_argument('--ner', type=str2bool, nargs='?', const=True, default=True,
                        help='use named entity tags as a feature.')
    
    parser.add_argument('--use_qemb', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--concat_rnn_layers', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--dropout_emb', type=float, default=0.4)
    parser.add_argument('--dropout_rnn', type=float, default=0.4)
    parser.add_argument('--dropout_rnn_output', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--max_len', type=int, default=15)


    args = parser.parse_args()

    # set model dir Create one if the model directory does not
    # exist previously.
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    args.model_dir = os.path.abspath(model_dir)


    # Set the random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # setup logger
    class ProgressHandler(logging.Handler):
        def __init__(self, level=logging.NOTSET):
            super().__init__(level)

        def emit(self, record):
            log_entry = self.format(record)
            if record.message.startswith('> '):
                sys.stdout.write('{}\r'.format(log_entry.rstrip()))
                sys.stdout.flush()
            else:
                sys.stdout.write('{}\n'.format(log_entry))

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(args.model_dir, 'log.txt'))
    fh.setLevel(logging.INFO)
    ch = ProgressHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)

    return args, log


def lr_decay(optimizer, lr_decay):
    # Decaying Learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer


def load_data(opt):
    with open('SQuAD/meta.msgpack', 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    opt['pretrained_words'] = True
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)
    opt['pos_size'] = len(meta['vocab_tag'])
    opt['ner_size'] = len(meta['vocab_ent'])
    BatchGen.pos_size = opt['pos_size']
    BatchGen.ner_size = opt['ner_size']
    with open(opt['data_file'], 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    train = data['train']
    data['dev'].sort(key=lambda x: len(x[1]))
    dev = [x[:-1] for x in data['dev']]
    dev_y = [x[-1] for x in data['dev']]
    return train, dev, dev_y, embedding, opt



class BatchGen:
    # Helper class for generating the batches.
    pos_size = None
    ner_size = None

    def __init__(self, data, batch_size, gpu, evaluation=False):
    
        self.batch_size = batch_size
        self.eval = evaluation
        self.gpu = gpu

        # sort by len
        data = sorted(data, key=lambda x: len(x[1]))
       
        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

        # Random shuffling of the data
        if not evaluation:
            random.shuffle(data)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        # Get the batch of data
        for batch in self.data:
            
            # Size of the batch
            batch_size = len(batch)
            batch = list(zip(*batch))
            if self.eval:
                assert len(batch) == 8
            else:
                assert len(batch) == 10

           
            context_len = max(len(x) for x in batch[1])
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[1]):
                context_id[i, :len(doc)] = torch.LongTensor(doc)

            feature_len = len(batch[2][0][0])

            # Exact Match between paragraph words and the question.
            context_feature = torch.Tensor(batch_size, context_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[2]):
                for j, feature in enumerate(doc):
                    context_feature[i, j, :] = torch.Tensor(feature)

            # Find the Parts of Speech Tag corresponding to the word and fill 
            # corresponding index with 1.
            context_tag = torch.Tensor(batch_size, context_len, self.pos_size).fill_(0)
            for i, doc in enumerate(batch[3]):
                for j, tag in enumerate(doc):
                    context_tag[i, j, tag] = 1
            
            # Find the Named Entity Recognition Tag and fill the corresponding index with 1
            context_ent = torch.Tensor(batch_size, context_len, self.ner_size).fill_(0)
            for i, doc in enumerate(batch[4]):
                for j, ent in enumerate(doc):
                    context_ent[i, j, ent] = 1


            question_len = max(len(x) for x in batch[5])
            question_id = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, doc in enumerate(batch[5]):
                question_id[i, :len(doc)] = torch.LongTensor(doc)

            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)
            text = list(batch[6])
            span = list(batch[7])
            if not self.eval:
                y_s = torch.LongTensor(batch[8])
                y_e = torch.LongTensor(batch[9])
            
            if self.gpu:
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory()
                context_tag = context_tag.pin_memory()
                context_ent = context_ent.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()


            if self.eval:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, text, span)
            else:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, y_s, y_e, text, span)


def _normalize_answer(s):

    def remove_articles(text):
        # Remove articles
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        # Remove extra white spaces
        return ' '.join(text.split())

    def remove_punc(text):
        # Remove the punctuations
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        # Convert the text to lowercase
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match(pred, answers):
    # Comparing the predicted output
    # with the ground truth.
    if pred is None or answers is None:
        return False
    pred = _normalize_answer(pred)
    for a in answers:
        if pred == _normalize_answer(a):
            return True
    return False


def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        # Compute the overlap between the tokens from the ground truth
        # and the predicted answer
        common = Counter(g_tokens) & Counter(a_tokens)
        # Number of tokens that are common between prediction and ground truth.
        num_same = sum(common.values())
        # If none are matching return 
        if num_same == 0:
            return 0
        # Compute the precision : true positives/(true positives + false positives)
        precision = 1. * num_same / len(g_tokens)
        # What proportion of true positives was identified correctly.
        recall = 1. * num_same / len(a_tokens)
        # Compute the F1 score which is the harmonic mean fo precision and recall.
        f1 = (2 * precision * recall) / (precision + recall)
        # Return the F1 score.
        return f1

    if pred is None or answers is None:
        return 0
    g_tokens = _normalize_answer(pred).split()
    scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers]
    # Obtain the scores from the predicted and ground truth answers.
    return max(scores)

# Compute the em score
def score(pred, truth):
    assert len(pred) == len(truth)
    
    f1 = em = total = 0
    for p, t in zip(pred, truth):
        total += 1
        em += _exact_match(p, t) # Check number of exact matches
        f1 += _f1_score(p, t) # F1 score
    
    # Compute EM and F1 Score
    em = 100. * em / total 
    f1 = 100. * f1 / total
    return em, f1


if __name__ == '__main__':
    main()


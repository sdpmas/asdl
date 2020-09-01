# coding=utf-8
from __future__ import print_function

import time

import astor
import six.moves.cPickle as pickle
from six.moves import input
from six.moves import xrange as range
from torch.autograd import Variable

import evaluation
from asdl.asdl import ASDLGrammar
from asdl.transition_system import TransitionSystem
from common.utils import update_args, init_arg_parser
from components.dataset_new import Dataset
from components.reranker import *
from components.standalone_parser import StandaloneParser
from model import nn_utils
from model.paraphrase import ParaphraseIdentificationModel
from model.new_parser import Parser
from model.reconstruction_model import Reconstructor
from model.utils import GloveHelper

# important, make sure the astor version matches here.
# assert astor.__version__ == "0.7.1"
if six.PY3:
    pass


def init_config():
    args = arg_parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args

def train(args):
    """Maximum Likelihood Estimation"""
    args.dropout=0.1
    args.hidden_size=128
    args.embed_size=128
    args.beam_size=1
    args.action_embed_size=128
    args.lr=0.001
    args.cuda=True if torch.cuda.is_available() else False
    args.decay_lr_every_epoch=True
    args.sup_attention=False
    # load in train/dev set
    train_set = Dataset.from_bin_file(args.train_file)

    if args.dev_file:
        dev_set = Dataset.from_bin_file(args.dev_file)
    else: dev_set = Dataset(examples=[])

    vocab = pickle.load(open(args.vocab, 'rb'))

    grammar = ASDLGrammar.from_text(open(args.asdl_file).read())
    transition_system = Registrable.by_name(args.transition_system)(grammar)
    print (args.action_embed_size,args.no_copy,args.dropout)
    parser_cls = Registrable.by_name(args.parser)  # TODO: add arg
    if args.pretrain:
        print('Finetune with: ', args.pretrain, file=sys.stderr)
        model = parser_cls.load(model_path=args.pretrain, cuda=args.cuda)
    else:
        model = parser_cls(args, vocab, transition_system)

    model.train()
    evaluator = Registrable.by_name(args.evaluator)(transition_system, args=args)
    if args.cuda: model.cuda()

    optimizer_cls = eval('torch.optim.%s' % args.optimizer)  # FIXME: this is evil!
    optimizer = optimizer_cls(model.parameters(), lr=args.lr)

    if not args.pretrain:
        if args.uniform_init:
            print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init), file=sys.stderr)
            nn_utils.uniform_init(-args.uniform_init, args.uniform_init, model.parameters())
        elif args.glorot_init:
            print('use glorot initialization', file=sys.stderr)
            nn_utils.glorot_init(model.parameters())

        # load pre-trained word embedding (optional)
        if args.glove_embed_path:
            print('load glove embedding from: %s' % args.glove_embed_path, file=sys.stderr)
            glove_embedding = GloveHelper(args.glove_embed_path)
            glove_embedding.load_to(model.src_embed, vocab.source)

    print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = report_sup_att_loss = 0.
    history_dev_scores = []
    num_trial = patience = 0
    print ('this is the new small one masked new 2 layers print new vocab final alternative final w w cuda new lr')
    while True:
        epoch += 1
        print ('lr for this epoch is ',optimizer.param_groups[0]['lr'])
        epoch_begin = time.time()

        for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
            batch_examples = [e for e in batch_examples if len(e.tgt_actions) <= args.decode_max_time_step]
            train_iter += 1
            optimizer.zero_grad()

            ret_val = model.score(batch_examples)
            loss = -ret_val[0]

            # print(loss.data)
            loss_val = torch.sum(loss).data.item()
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)

            if args.sup_attention:
                att_probs = ret_val[1]
                if att_probs:
                    sup_att_loss = -torch.log(torch.cat(att_probs)).mean()
                    sup_att_loss_val = sup_att_loss.data[0]
                    report_sup_att_loss += sup_att_loss_val

                    loss += sup_att_loss

            loss.backward()

            # clip gradient
            if args.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

            if train_iter % args.log_every == 0:
                log_str = '[Iter %d] encoder loss=%.5f' % (train_iter, report_loss / report_examples)
                if args.sup_attention:
                    log_str += ' supervised attention loss=%.5f' % (report_sup_att_loss / report_examples)
                    report_sup_att_loss = 0.

                print(log_str, file=sys.stderr)
                report_loss = report_examples = 0.

        if args.decay_lr_every_epoch and epoch > 0:
            lr = optimizer.param_groups[0]['lr'] * 0.7
            print('decay learning rate to %f' % lr, file=sys.stderr)

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

        if args.save_all_models:
            model_file = args.save_to + '.iter%d.bin' % train_iter
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)
        if epoch>0:
            if args.dev_file:
                if epoch % args.valid_every_epoch == 0:
                    print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
                    eval_start = time.time()
                    eval_results = evaluation.evaluate(dev_set.examples, model, evaluator, args,
                                                    verbose=False, eval_top_pred_only=args.eval_top_pred_only)
                    # dev_score = eval_results[evaluator.default_metric]
                    dev_score=eval_results

                    print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (
                                        epoch, eval_results,
                                        evaluator.default_metric,
                                        dev_score,
                                        time.time() - eval_start), file=sys.stderr)

                    is_better = history_dev_scores == [] or dev_score > max(history_dev_scores)
                    history_dev_scores.append(dev_score)


if __name__ == '__main__':
    arg_parser = init_arg_parser()
    args = init_config()
    print(args, file=sys.stderr)
    if args.mode == 'train':
        train(args)
    elif args.mode in ('train_reconstructor', 'train_paraphrase_identifier'):
        train_rerank_feature(args)
    elif args.mode == 'rerank':
        train_reranker_and_test(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'interactive':
        interactive_mode(args)
    else:
        raise RuntimeError('unknown mode')

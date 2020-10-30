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
    args.hidden_size=512
    args.embed_size=512
    args.beam_size=1
    args.action_embed_size=512
    args.field_embed_size=64
    args.type_embed_size=64 
    args.lr=3e-5
    args.max_epoch=50
    args.cuda=True if torch.cuda.is_available() else False
    # args.cuda=False
    args.decay_lr_every_epoch=True
    args.sup_attention=False
    # load in train/dev set
    train_set = Dataset.from_bin_file(args.train_file)
    # train_set = Dataset.from_bin_file_joint('data/conala/src.bin','data/conala/pretrain.bin')
    print (len(train_set),'this is the length of train set')
    if args.dev_file:
        dev_set = Dataset.from_bin_file_test(args.test_file)
    else: dev_set = Dataset(examples=[])

    vocab = pickle.load(open(args.vocab, 'rb'))
    vocab_src=vocab

    grammar = ASDLGrammar.from_text(open(args.asdl_file).read())
    transition_system = Registrable.by_name(args.transition_system)(grammar)
    print (args.action_embed_size,args.no_copy,args.dropout)
    parser_cls = Registrable.by_name(args.parser)  # TODO: add arg
    if args.pretrain:
        print('Finetune with: ', args.pretrain, file=sys.stderr)
        model = parser_cls.load(model_path=args.pretrain, cuda=args.cuda)
    else:
        model = parser_cls(args, vocab, transition_system,src_vocab=vocab)

    model.train()
    evaluator = Registrable.by_name(args.evaluator)(transition_system, args=args)
    if args.cuda: model.cuda()

    optimizer_cls = eval('torch.optim.%s' % args.optimizer)
    optimizer = optimizer_cls(model.parameters(), lr=args.lr)
    # if args.mass_pretrain:
    #    nn_utils.glorot_init(model.parameters()) 
    # if not args.pretrain:
    #     if args.uniform_init:
    #         print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init), file=sys.stderr)
    #         nn_utils.uniform_init(-args.uniform_init, args.uniform_init, model.parameters())
    #     elif args.glorot_init:
    #         print('use glorot initialization', file=sys.stderr)
    #         # nn_utils.glorot_init(model.parameters())

        

    print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)
    print('vocab: %s' % repr(vocab_src), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = report_sup_att_loss = 0.
    history_dev_scores = []
    num_trial = patience = 0
    
    while True:
        epoch += 1
        print ('lr for this epoch is ',optimizer.param_groups[0]['lr'],' and the current epoch is ',epoch)
        epoch_begin = time.time()

        for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
            batch_examples = [e for e in batch_examples if len(e.tgt_actions) <= args.decode_max_time_step]
            train_iter += 1
            optimizer.zero_grad()
            # print ('it gets here')
            if args.mass_pretrain:
                # print ('mass pretrain is true')
                ret_val=model.score(batch_examples, pretrain=True)
            else:
                # print ('it gets here')
                ret_val = model.score(batch_examples,pretrain=False)

            loss = -ret_val[0]

            # print(loss.data)
            loss_val = torch.sum(loss).data.item()
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)

            

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

        if args.decay_lr_every_epoch and epoch >10:
            lr = optimizer.param_groups[0]['lr'] * 0.95
            print('decay learning rate to %f' % lr, file=sys.stderr)

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

        # if args.save_all_models:
        #     model_file = args.save_to + '.iter%d.bin' % train_iter
        #     print('save model to [%s]' % model_file, file=sys.stderr)
        #     model.save(model_file)
        # if epoch>5:
        if not args.mass_pretrain and epoch>32:
            is_better = False
            if args.dev_file:
                print ('dev file validation')

                print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
                eval_start = time.time()
                eval_results = evaluation.evaluate(dev_set.examples, model, evaluator, args,
                                                verbose=False, eval_top_pred_only=args.eval_top_pred_only)
                # dev_score = eval_results[evaluator.default_metric]
                # dev_score=eval_results

                # for batch_examples in dev_set.batch_iter(batch_size=args.batch_size, shuffle=True):
                #     batch_examples = [e for e in batch_examples if len(e.tgt_actions) <= args.decode_max_time_step]
                #     # train_iter += 1
                #     # optimizer.zero_grad()
                #     model.eval()
                #     ret_val = model.score(batch_examples)
                #     loss = -ret_val[0]

                #     print ('the valid loss is: ',loss.mean())
                print ('samip dahal')
                # print('[Epoch %d] evaluate details: %s, dev %s: %.5f ' % (
                #                     epoch, eval_results,
                #                     evaluator.default_metric,
                #                     dev_score))
                # print("Evaluation ",epoch, eval_results,dev_score)
                dev_score = eval_results[evaluator.default_metric]

                print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (
                                    epoch, eval_results,
                                    evaluator.default_metric,
                                    dev_score,
                                    time.time() - eval_start), file=sys.stderr)
                is_better = history_dev_scores == [] or dev_score > max(history_dev_scores)
                history_dev_scores.append(dev_score)
                model.train()
            else:
                is_better = True
            if is_better :
                print("the not mentioned tags are ",model.new_tags)
                patience = 0
                model_file = f'saved_models/pre_small/{epoch}.bin'
                print (model_file)
                if history_dev_scores:
                    s=history_dev_scores[-1]
                    model_file = f'saved_models/pre_small/{epoch}${s}.bin'
                print('save the current model ..', file=sys.stderr)
                print('save model to [%s]' % model_file, file=sys.stderr)
                model.save(model_file)
                # also save the optimizers' state
                torch.save(optimizer.state_dict(), model_file + '.optim.bin')
        else:
            print ('it is mass')
            if epoch%5==0 and args.mass_pretrain:
                model_file = f'saved_models/code_fine_4/{epoch}{loss}.bin'
                print('save the current model ..', file=sys.stderr)
                print('save model to [%s]' % model_file, file=sys.stderr)
                model.save(model_file)
                # also save the optimizers' state
                torch.save(optimizer.state_dict(), f'saved_models/code_fine_4/{epoch}{loss}' + '.optim.bin')
        
        if epoch == args.max_epoch:
            print('reached max epoch, stop!', file=sys.stderr)
            exit(0)

def test(args):
    test_set = Dataset.from_bin_file(args.test_file)
    assert args.load_model

    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model)
    transition_system = params['transition_system']
    saved_args = params['args']
    saved_args.cuda = args.cuda
    # set the correct domain from saved arg
    args.lang = saved_args.lang

    parser_cls = Registrable.by_name(args.parser)
    parser = parser_cls.load(model_path=args.load_model, cuda=args.cuda)
    # print("the depthe embed weight: ",parser.production_embed.weight)
    # print("the depthe embed weight: ",parser.depth_embed.weight)
    print(len(list(parser.parameters())),'this is the tot params')
    parser.eval()
    evaluator = Registrable.by_name(args.evaluator)(transition_system, args=args)
    eval_results, decode_results = evaluation.evaluate(test_set.examples, parser, evaluator, args,
                                                       verbose=args.verbose, return_decode_result=True)
    print(eval_results, file=sys.stderr)
    # targets=[]
    # for i,ex in enumerate(examples):
    #     targets.append(ex.meta['example_dict']['snippet'])

    if args.save_decode_to:
        pickle.dump(decode_results, open(args.save_decode_to, 'wb'))
        # pickle.dump(targets, open('decodes/conala/actual', 'wb'))

def train_reranker_and_test(args):
    # args.no_copy=True
    print('load dataset [test %s], [dev %s]' % (args.test_file, args.dev_file), file=sys.stderr)
    test_set = Dataset.from_bin_file(args.test_file)
    dev_set = Dataset.from_bin_file(args.dev_file)

    features = []
    i = 0
    while i < len(args.features):
        feat_name = args.features[i]
        feat_cls = Registrable.by_name(feat_name)
        print('Add feature %s' % feat_name, file=sys.stderr)
        if issubclass(feat_cls, nn.Module):
            feat_path = os.path.join('saved_models/conala/', args.features[i] + '.bin')
            feat_inst = feat_cls.load(feat_path)
            print('Load feature %s from %s' % (feat_name, feat_path), file=sys.stderr)
        else:
            feat_inst = feat_cls()

        features.append(feat_inst)
        i += 1

    transition_system = next(feat.transition_system for feat in features if hasattr(feat, 'transition_system'))
    evaluator = Registrable.by_name(args.evaluator)(transition_system)


    print('load dev decode results [%s]' % args.dev_decode_file, file=sys.stderr)
    dev_decode_results = pickle.load(open(args.dev_decode_file, 'rb'))
    dev_eval_results = evaluator.evaluate_dataset(dev_set.examples, dev_decode_results, fast_mode=False)

    print('load test decode results [%s]' % args.test_decode_file, file=sys.stderr)
    test_decode_results = pickle.load(open(args.test_decode_file, 'rb'))
    test_eval_results = evaluator.evaluate_dataset(test_set.examples, test_decode_results, fast_mode=False)

    print('Dev Eval Results', file=sys.stderr)
    print(dev_eval_results, file=sys.stderr)
    print('Test Eval Results', file=sys.stderr)
    print(test_eval_results, file=sys.stderr)

    if args.load_reranker:
        reranker = GridSearchReranker.load(args.load_reranker)
    else:
        print('we are training the reranker')
        reranker = GridSearchReranker(features, transition_system=transition_system)
        args.num_workers=1
        if args.num_workers == 1:
            reranker.train(dev_set.examples, dev_decode_results, evaluator=evaluator)
            print('we are over this phase')
        else:
            print('multi process')
            reranker.train_multiprocess(dev_set.examples, dev_decode_results, evaluator=evaluator, num_workers=args.num_workers)
            print('we are over this phase')

        if args.save_to:
            print('Save Reranker to %s' % args.save_to, file=sys.stderr)
            reranker.save(args.save_to)

    test_score_with_rerank = reranker.compute_rerank_performance(test_set.examples, test_decode_results, 
                                                                 evaluator=evaluator, args=args)
    
    dev_score_with_rerank = reranker.compute_rerank_performance(dev_set.examples, dev_decode_results, 
                                                                 evaluator=evaluator, args=args)
    print('Test Eval Results After Reranking', file=sys.stderr)
    print(test_score_with_rerank, file=sys.stderr)
    print(dev_score_with_rerank)


def train_rerank_feature(args):
    train_set = Dataset.from_bin_file(args.train_file)
    dev_set = Dataset.from_bin_file(args.dev_file)
    vocab = pickle.load(open(args.vocab, 'rb'))

    grammar = ASDLGrammar.from_text(open(args.asdl_file).read())
    transition_system = TransitionSystem.get_class_by_lang(args.lang)(grammar)

    train_paraphrase_model = args.mode == 'train_paraphrase_identifier'

    def _get_feat_class():
        if args.mode == 'train_reconstructor':
            return Reconstructor
        elif args.mode == 'train_paraphrase_identifier':
            return ParaphraseIdentificationModel

    def _filter_hyps(_decode_results):
        for i in range(len(_decode_results)):
            valid_hyps = []
            for hyp in _decode_results[i]:
                try:
                    transition_system.tokenize_code(hyp.code)
                    valid_hyps.append(hyp)
                except: pass

            _decode_results[i] = valid_hyps

    model = _get_feat_class()(args, vocab, transition_system)

    if args.glorot_init:
        print('use glorot initialization', file=sys.stderr)
        nn_utils.glorot_init(model.parameters())

    model.train()
    if args.cuda: model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # if training the paraphrase model, also load in decoding results
    if train_paraphrase_model:
        print('load training decode results [%s]' % args.train_decode_file, file=sys.stderr)
        train_decode_results = pickle.load(open(args.train_decode_file, 'rb'))
        _filter_hyps(train_decode_results)
        train_decode_results = {e.idx: hyps for e, hyps in zip(train_set, train_decode_results)}

        print('load dev decode results [%s]' % args.dev_decode_file, file=sys.stderr)
        dev_decode_results = pickle.load(open(args.dev_decode_file, 'rb'))
        _filter_hyps(dev_decode_results)
        dev_decode_results = {e.idx: hyps for e, hyps in zip(dev_set, dev_decode_results)}

    def evaluate_ppl():
        model.eval()
        cum_loss = 0.
        cum_tgt_words = 0.
        for batch in dev_set.batch_iter(args.batch_size):
            loss = -model.score(batch).sum()
            cum_loss += loss.data.item()
            cum_tgt_words += sum(len(e.src_sent) + 1 for e in batch)  # add ending </s>

        ppl = np.exp(cum_loss / cum_tgt_words)
        model.train()
        return ppl

    def evaluate_paraphrase_acc():
        model.eval()
        labels = []
        for batch in dev_set.batch_iter(args.batch_size):
            probs = model.score(batch).exp().data.cpu().numpy()
            for p in probs:
                labels.append(p >= 0.5)

            # get negative examples
            batch_decoding_results = [dev_decode_results[e.idx] for e in batch]
            batch_negative_examples = [get_negative_example(e, _hyps, type='best')
                                       for e, _hyps in zip(batch, batch_decoding_results)]
            batch_negative_examples = list(filter(None, batch_negative_examples))
            probs = model.score(batch_negative_examples).exp().data.cpu().numpy()
            for p in probs:
                labels.append(p < 0.5)

        acc = np.average(labels)
        model.train()
        return acc

    def get_negative_example(_example, _hyps, type='sample'):
        incorrect_hyps = [hyp for hyp in _hyps if not hyp.is_correct]
        if incorrect_hyps:
            incorrect_hyp_scores = [hyp.score for hyp in incorrect_hyps]
            if type in ('best', 'sample'):
                if type == 'best':
                    sample_idx = np.argmax(incorrect_hyp_scores)
                    sampled_hyp = incorrect_hyps[sample_idx]
                else:
                    incorrect_hyp_probs = [np.exp(score) for score in incorrect_hyp_scores]
                    incorrect_hyp_probs = np.array(incorrect_hyp_probs) / sum(incorrect_hyp_probs)
                    sampled_hyp = np.random.choice(incorrect_hyps, size=1, p=incorrect_hyp_probs)
                    sampled_hyp = sampled_hyp[0]

                sample = Example(idx='negative-%s' % _example.idx,
                                 src_sent=_example.src_sent,
                                 tgt_code=sampled_hyp.code,
                                 tgt_actions=None,
                                 tgt_ast=None)
                return sample
            elif type == 'all':
                samples = []
                for i, hyp in enumerate(incorrect_hyps):
                    sample = Example(idx='negative-%s-%d' % (_example.idx, i),
                                     src_sent=_example.src_sent,
                                     tgt_code=hyp.code,
                                     tgt_actions=None,
                                     tgt_ast=None)
                    samples.append(sample)

                return samples
        else:
            return None

    print('begin training decoder, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = 0.
    history_dev_scores = []
    num_trial = patience = 0
    while True:
        epoch += 1
        epoch_begin = time.time()

        for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
            batch_examples = [e for e in batch_examples if len(e.tgt_actions) <= args.decode_max_time_step]

            if train_paraphrase_model:
                positive_examples_num = len(batch_examples)
                labels = [0] * len(batch_examples)
                negative_samples = []
                batch_decoding_results = [train_decode_results[e.idx] for e in batch_examples]
                # sample negative examples
                for example, hyps in zip(batch_examples, batch_decoding_results):
                    if hyps:
                        negative_sample = get_negative_example(example, hyps, type=args.negative_sample_type)
                        if negative_sample:
                            if isinstance(negative_sample, Example):
                                negative_samples.append(negative_sample)
                                labels.append(1)
                            else:
                                negative_samples.extend(negative_sample)
                                labels.extend([1] * len(negative_sample))

                batch_examples += negative_samples

            train_iter += 1
            optimizer.zero_grad()

            nll = -model(batch_examples)
            if train_paraphrase_model:
                idx_tensor = Variable(torch.LongTensor(labels).unsqueeze(-1), requires_grad=False)
                if args.cuda: idx_tensor = idx_tensor.cuda()
                loss = torch.gather(nll, 1, idx_tensor)
            else:
                loss = nll

            # print(loss.data)
            loss_val = torch.sum(loss).data.item()
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

            if train_iter % args.log_every == 0:
                print('[Iter %d] encoder loss=%.5f' %
                      (train_iter,
                       report_loss / report_examples),
                      file=sys.stderr)

                report_loss = report_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

        # perform validation
        print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
        eval_start = time.time()
        # evaluate dev_score
        dev_acc = evaluate_paraphrase_acc() if train_paraphrase_model else -evaluate_ppl()
        print('[Epoch %d] dev_score=%.5f took %ds' % (epoch, dev_acc, time.time() - eval_start), file=sys.stderr)
        is_better = history_dev_scores == [] or dev_acc > max(history_dev_scores)
        history_dev_scores.append(dev_acc)

        if is_better:
            patience = 0
            model_file = args.save_to + '.bin'
            print('save currently the best model ..', file=sys.stderr)
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), args.save_to + '.optim.bin')
        elif patience < args.patience:
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

        if patience == args.patience:
            num_trial += 1
            print('hit #%d trial' % num_trial, file=sys.stderr)
            if num_trial == args.max_num_trial:
                print('early stop!', file=sys.stderr)
                exit(0)

            # decay lr, and restore from previously best checkpoint
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

            # load model
            params = torch.load(args.save_to + '.bin', map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            if args.cuda: model = model.cuda()

            # load optimizers
            if args.reset_optimizer:
                print('reset optimizer', file=sys.stderr)
                optimizer = torch.optim.Adam(model.inference_model.parameters(), lr=lr)
            else:
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(args.save_to + '.optim.bin'))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0

if __name__ == '__main__':
    arg_parser = init_arg_parser()
    args = init_config()
    print(args, file=sys.stderr)
    if args.mode == 'train':
        train(args)
    elif args.mode in ('train_reconstructor', 'train_paraphrase_identifier'):
        train_rerank_feature(args)
    elif args.mode=='rerank':
        train_reranker_and_test(args)
    elif args.mode == 'test':
        test(args)

    else:
        raise RuntimeError('unknown mode')

# coding=utf-8
from __future__ import print_function

import os
from six.moves import xrange as range
import math
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from fastai.text import *

from asdl.hypothesis import *
from asdl.transition_system import *
from common.registerable import Registrable
from components.decode_hypothesis import DecodeHypothesis
from components.action_info import *
from components.dataset_new import Batch
from common.utils import update_args, init_arg_parser
from model import nn_utils
from model.attention_util import AttentionUtil
from model.pointer_net import PointerNet


from fastai.text import *





@Registrable.register('default_parser')
class Parser(nn.Module):
    def __init__(self, args, vocab, transition_system, n_layers=2, n_heads=8, d_head=None,
                 d_inner=1024, p=0.1, bias=True, scale=True, double_drop=True, pad_idx=0):
        super(Parser, self).__init__()
        # scripts/conala/new_train.sh

        self.args = args
        print(self.args) 
        self.args.no_copy=False
        args.no_copy=False
        self.vocab = vocab
        self.input_dim=args.action_embed_size 
        self.input_dim += args.action_embed_size 
        self.input_dim += args.field_embed_size 
        self.input_dim += args.type_embed_size

        self.transition_system = transition_system
        self.grammar = self.transition_system.grammar

        
        self.src_embed = nn.Embedding(len(vocab.source), args.embed_size)
        self.mask_embed=nn.Embedding(1,args.embed_size)
        self.sep_embed=nn.Embedding(1,args.embed_size)

        
        self.production_embed = nn.Embedding(len(transition_system.grammar) + 1, args.action_embed_size)

        
        self.primitive_embed = nn.Embedding(len(vocab.primitive), args.action_embed_size)

        
        self.field_embed = nn.Embedding(len(transition_system.grammar.fields), args.field_embed_size)

        
        self.type_embed = nn.Embedding(len(transition_system.grammar.types), args.type_embed_size)

        

        
        args_new = (n_heads, args.embed_size, d_head, args.embed_size, p, bias, scale, double_drop)
        args_new_dec = (n_heads, self.input_dim, d_head, self.input_dim, p, bias, scale, double_drop)
        self.encoder = nn.ModuleList([EncoderBlock(*args_new) for _ in range(n_layers)])
        # self.encoder=EncoderBlock(*args_new)
        self.ende_inp=nn.Linear(args.action_embed_size,self.input_dim)
        self.decoder = nn.ModuleList([DecoderBlock(*args_new_dec) for _ in range(n_layers)])
        # self.decoder =DecoderBlock(*args_new_dec)
        self.out = nn.Linear(self.input_dim, args.action_embed_size)
        #TODO: adjust the dimensions
        self.src_pointer_net = PointerNet(query_vec_size=args.action_embed_size, src_encoding_size=self.input_dim)
        self.primitive_predictor = nn.Linear(args.action_embed_size, 2)
        self.pad_idx = pad_idx
        #TODO: apply bias
       
        self.query_vec_to_action_embed = nn.Linear(args.action_embed_size, args.embed_size, bias=args.readout == 'non_linear')
               
        self.query_vec_to_primitive_embed = nn.Linear(args.action_embed_size, args.embed_size, bias=args.readout == 'non_linear')
        self.query_vec_to_primitive_embed = self.query_vec_to_action_embed

        self.read_out_act = torch.tanh 
        

        self.production_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_action_embed(q)),
                                                         self.production_embed.weight)
        self.tgt_token_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_primitive_embed(q)),
                                                        self.primitive_embed.weight)
        self.tgt_text_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_primitive_embed(q)),
                                                        self.src_embed.weight)
        self.dropout = nn.Dropout(args.dropout)
        

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
            # self.t=torch.cuda.Tensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor
        self.zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())
        #looks all good

    def step(self, batch):
        inp=[]
        for example in batch.examples:
            
            inp_row=[]
            # inp_row.append(Variable(self.new_tensor(self.args.action_embed_size).zero_()))
            for t in range(max(batch.src_sents_len)):
                
                if t<len(example.input_actions):
                    a_tm1=example.input_actions[t]
                    
                    if isinstance(a_tm1, ApplyRuleAction):
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[a_tm1.production]]
                    elif isinstance(a_tm1, ReduceAction):
                        a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                    elif isinstance(a_tm1, GenTextAction):
                        a_tm1_embed=self.src_embed.weight[self.vocab.source[a_tm1.text]]
                    # elif isinstance(a_tm1, SepAction):
                    #     a_tm1_embed=self.sep_embed.weight[0]
                    elif isinstance(a_tm1, MaskAction):
                        a_tm1_embed=self.mask_embed.weight[0]
                    else:
                        # print ('we have reached gen')
                        # print (a_tm1.mask)
                       
                        a_tm1_embed = self.primitive_embed.weight[self.vocab.primitive[a_tm1.token]]

                else:
                     a_tm1_embed =  Variable(self.new_tensor(self.args.action_embed_size).zero_())
                inp_row.append(a_tm1_embed)

            inp.append(torch.stack(inp_row))
        
        inp=torch.stack(inp)

        out=[]
        for example in batch.examples:
            
            out_row=[]
            # out_row.append(Variable(self.new_tensor(self.args.action_embed_size).zero_()))
            x=Variable(self.new_tensor(self.input_dim).zero_(),requires_grad=False)
            offset = self.args.action_embed_size  # prev_action
            offset += self.args.action_embed_size
            offset += self.args.field_embed_size
            
            # print (self.grammar.type2id[self.grammar.root_type])
            # print (offset, self.input_dim)
            # print (self.type_embed.weight[self.grammar.type2id[self.grammar.root_type]].shape)
            x[offset: offset + self.args.type_embed_size] = self.type_embed.weight[self.grammar.type2id[self.grammar.root_type]]
            out_row.append(x)
            for t in range(batch.max_action_num-1):
                embeds=[]
                if t<len(example.tgt_actions):
                    a_tm1=example.tgt_actions[t].action
                    
                        # print (isinstance(a_tm1, MaskAction))
                        # print (a_tm1.__name__)
                        # print (a_tm1)
                    if isinstance(a_tm1, ApplyRuleAction):
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[a_tm1.production]]
                    elif isinstance(a_tm1, ReduceAction):
                        a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                    elif isinstance(a_tm1, GenTextAction):
                        a_tm1_embed=self.src_embed.weight[self.vocab.source[a_tm1.text]]
                    # elif isinstance(a_tm1, SepAction):
                    #     a_tm1_embed=self.sep_embed.weight[0]
                    elif isinstance(a_tm1, MaskAction):
                        a_tm1_embed=self.mask_embed.weight[0]
                    else:
                        # print ('we have reached gen')
                        # print (a_tm1.mask)
                       
                        a_tm1_embed = self.primitive_embed.weight[self.vocab.primitive[a_tm1.token]]

                else:
                     a_tm1_embed =  Variable(self.new_tensor(self.args.action_embed_size).zero_())
                embeds.append(a_tm1_embed)
                # xyz=torch.tensor(batch.frontier_prod_idx(example,t+1))
                # print (xyz)
                parent_production_embed = self.production_embed.weight[torch.tensor(batch.frontier_prod_idx(example,t+1))]
                embeds.append(parent_production_embed)
                parent_field_embed = self.field_embed.weight[torch.tensor(batch.frontier_field_idx(example,t+1))]
                embeds.append(parent_field_embed)
                parent_field_type_embed = self.type_embed.weight[torch.tensor(batch.frontier_field_type_idx(example,t+1))]
                embeds.append(parent_field_type_embed)

                embeds=torch.cat(embeds,dim=-1)
                # print (embeds.shape)
                out_row.append(embeds)



                # out_row.append(a_tm1_embed)

            out.append(torch.stack(out_row))
        
        out1=torch.stack(out)
        # print (inp.shape,out1.shape)
        # print (out1.shape)
        # print (batch.max_action_num)
        # inp=self.new_tensor(torch.randn(inp.shape[0],inp.shape[1],inp.shape[2]))

        mask_out = get_output_mask(out1, self.pad_idx)
        enc = compose(self.encoder)(inp)
        # enc=self.encoder(inp)
        enc=self.ende_inp(enc)
        out = compose(self.decoder)(out1, enc,mask_out)
        # out=self.decoder(out1, enc,mask_out)
        # print ('decoder done')
        out=self.out(out)
        # print ('out done',out.shape)
        return out,enc

    

    def accuracy(self, pred, target,mask):
        pred1=torch.argmax(pred, dim=-1).squeeze()
        tot_count=0.
        tot_correct=0.
        for p,t,mask1 in zip(pred1, target.squeeze(),mask.squeeze()):
            tot_correct+=torch.sum(((p==t).float())*mask1)
            for i in t:
                if i!=torch.tensor(0):
                    tot_count+=1

        return tot_correct/tot_count
    

    def score(self,examples):
        batch = Batch(examples, self.grammar, self.vocab, copy=self.args.no_copy is False, cuda=self.args.cuda)
        
        with torch.no_grad():
            query_vectors,src_encodings=self.step(batch)
        # print (query_vectors.shape)
        apply_rule_prob = F.softmax(self.production_readout(query_vectors), dim=-1)
        # print (apply_rule_prob.shape)
        # print (batch.apply_rule_idx_matrix.unsqueeze(dim=2).shape)
        tgt_apply_rule_prob = torch.gather(apply_rule_prob, dim=2,
                                           index=batch.apply_rule_idx_matrix.unsqueeze(dim=2)).squeeze(2)
 
       
        action_acc=self.accuracy(apply_rule_prob,batch.apply_rule_idx_matrix.unsqueeze(dim=2),batch.apply_rule_mask)
        
       
        # print (query_vectors.shape)
        gen_from_vocab_prob = F.softmax(self.tgt_token_readout(query_vectors), dim=-1)
        # print (gen_from_vocab_prob.shape)
        token_acc=self.accuracy(gen_from_vocab_prob,batch.primitive_idx_matrix.unsqueeze(dim=2),batch.gen_token_mask)
       
        tgt_primitive_gen_from_vocab_prob = torch.gather(gen_from_vocab_prob, dim=2,
                                                         index=batch.primitive_idx_matrix.unsqueeze(dim=2)).squeeze(2)

        
        if not self.args.no_copy:
            primitive_predictor = F.softmax(self.primitive_predictor(query_vectors), dim=-1)
            primitive_copy_prob = self.src_pointer_net(src_encodings, batch.src_token_mask, query_vectors)
            #TODO: mask
            # print (query_vectors.shape)
            # print (primitive_predictor.shape)
            # print (src_encodings.shape)
            # print (primitive_copy_prob.shape,batch.primitive_copy_token_idx_mask.shape)
            tgt_primitive_copy_prob = torch.sum(primitive_copy_prob * batch.primitive_copy_token_idx_mask, dim=-1)
            action_mask_pad = torch.eq(batch.apply_rule_mask + batch.gen_token_mask + batch.primitive_copy_mask, 0.)
            action_mask = 1. - action_mask_pad.float()
            # print (primitive_predictor[:, :, 0].shape,gen_from_vocab_prob.shape, batch.gen_token_mask.shape)
            action_prob = tgt_apply_rule_prob * batch.apply_rule_mask + \
                          primitive_predictor[:, :, 0] * tgt_primitive_gen_from_vocab_prob * batch.gen_token_mask + \
                          primitive_predictor[:, :, 1] * tgt_primitive_copy_prob * batch.primitive_copy_mask
            action_prob.data.masked_fill_(action_mask_pad.data, 1.e-7)

            action_prob = action_prob.log() * action_mask
        else:
            print ('not working')
            tgt_primitive_gen_from_vocab_log_prob = tgt_primitive_gen_from_vocab_prob.log()
            action_prob = tgt_apply_rule_prob.log() * batch.apply_rule_mask +\
                          tgt_primitive_gen_from_vocab_log_prob * batch.gen_token_mask

        # txt_prob=F.softmax(self.tgt_text_readout(query_vectors), dim=-1)
        # txt_acc=self.accuracy(txt_prob,batch.text_idx_matrix.unsqueeze(dim=2),batch.gen_text_mask)
        # tgt_txt_prob = (torch.gather(txt_prob, dim=2,
                                    # index=batch.text_idx_matrix.unsqueeze(dim=2)).squeeze(2)).log()

            
       
                        #       tgt_txt_prob*batch.gen_text_mask

        scores = torch.sum(action_prob, dim=1)
        #score is of size batch_size
        print ('the accuracy for this batch is: ','action: ',action_acc,' token: ',token_acc)
        returns = [scores]
        return returns

    def parse(self, src_sent, context=None, beam_size=1, debug=False):
        # print ('beam_size',beam_size)
        beam_size=1
        # scripts/conala/train_retrieved_distsmpl.sh 100
        hyp_id=0
        args = self.args
        # self.args.no_copy=True
        primitive_vocab = self.vocab.primitive
        T = torch.cuda if args.cuda else torch
        # enc=[MaskAction()]
        a = 0
        hypothesis = DecodeHypothesis()
        completed_hypotheses = []
        # out_actions=[]
        enc=self.transition_system.get_actions(text=src_sent,grammar=self.grammar)
        # print (src_sent)
        # print (enc)
        # print (enc)
        # mask_out = get_output_mask(out1, self.pad_idx)
        inp=[]
        
        for t in range(len(enc)):
                 
            a_tm1=enc[t]
            
            if isinstance(a_tm1, ApplyRuleAction):
                a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[a_tm1.production]]
            elif isinstance(a_tm1, ReduceAction):
                a_tm1_embed = self.production_embed.weight[len(self.grammar)]
            elif isinstance(a_tm1, GenTextAction):
                a_tm1_embed=self.src_embed.weight[self.vocab.source[a_tm1.text]]
            # elif isinstance(a_tm1, SepAction):
            #     a_tm1_embed=self.sep_embed.weight[0]
            elif isinstance(a_tm1, MaskAction):
                # a_tm1_embed=self.mask_embed.weight[0]
                a_tm1_embed=self.zero_action_embed
            else:
                # print ('we have reached gen')
                # print (a_tm1.mask)
                # assert a_tm1==<class 'asdl.transition_system.MaskAction'>
                a_tm1_embed = self.primitive_embed.weight[self.vocab.primitive[a_tm1.token]]

            inp.append(a_tm1_embed)
        
        inp=torch.stack(inp).unsqueeze(dim=0)

        aggregated_primitive_tokens = OrderedDict()
        for token_pos, token in enumerate(src_sent):
            aggregated_primitive_tokens.setdefault(token, []).append(token_pos)
        
        src_encodings = compose(self.encoder)(inp)
        # src_encodings=self.encoder(inp)
        src_encodings=self.ende_inp(src_encodings)
        # print (src_encodings.shape,'src_encodings')
        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())
        
        out=[]
        out_running=[]
        while len(completed_hypotheses) < 1 and a<200 :
           
            if a==0:
                # print ('first')
                with torch.no_grad():
                    embed=Variable(self.new_tensor(self.input_dim).zero_())
                offset = self.args.action_embed_size  # prev_action
                offset += self.args.action_embed_size
                offset += self.args.field_embed_size
                embed[offset: offset + self.args.type_embed_size] = self.type_embed.weight[self.grammar.type2id[self.grammar.root_type]]
                out.append(embed) 
                out_running=out.copy()
                x=torch.stack(out_running).unsqueeze(dim=0)
            else:
                a_tm1 = hypothesis.actions[-1]
                # assert c_hyp.actions[-1]==hypothesis.actions[-1]
                # a_tm1_embeds = 
                # inp_running=inp_out.copy()
                
                if a_tm1:
                    embeds=[]
                    if isinstance(a_tm1, ApplyRuleAction):
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[a_tm1.production]]
                            #one dimensional
                    elif isinstance(a_tm1, ReduceAction):
                        a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                    else:
                        a_tm1_embed = self.primitive_embed.weight[self.vocab.primitive[a_tm1.token]]
                    # inp_out.append(a_tm1_embed)
                    # out_running=inp_out.copy()
                    # for i in range(1):
                    # print (type(out))
                    embeds.append(a_tm1_embed)
                    frontier_prod_embed = self.production_embed.weight[torch.tensor(\
                        self.grammar.prod2id[hypothesis.frontier_node.production])]
                    # parent_production_embed = self.production_embed.weight[torch.tensor(batch.frontier_prod_idx(example,t+1))]
                    embeds.append(frontier_prod_embed)
                    frontier_field_embed = self.field_embed.weight[torch.tensor(\
                        self.grammar.field2id[hypothesis.frontier_field.field])]
                    # parent_field_embed = self.field_embed.weight[torch.tensor(batch.frontier_field_idx(example,t+1))]
                    embeds.append(frontier_field_embed)
                    frontier_field_type = self.type_embed.weight[torch.tensor(\
                        self.grammar.type2id[hypothesis.frontier_field.type])]
                    embeds.append(frontier_field_type)

                    embeds=torch.cat(embeds,dim=-1)
                    out.append(embeds)
                    out_running=out.copy()
                    # out_running.append(zero_action_embed)
                        # inp_running.append(self.zero_action_embed)
                    
                        # a_tm1_embed_merged=[y,a_tm1_embed]
                    # a_tm1_embeds.append(torch.stack(out_actions))
                else:
                    print ('what"s happening')
                    out.append(Variable(self.new_tensor(self.input_dim).zero_()))
                    out_running=out.copy()
                    # out_running.append(zero_action_embed)
                        
                x=torch.stack(out_running).unsqueeze(dim=0)
                # print (x.shape)
                #gotta complete it in a while
                # x=a_tm1_embeds
            # print ('shape of input to the step_infer',x.shape)
            # mask_out = get_output_mask(x, self.pad_idx)
            out1 = compose(self.decoder)(x, src_encodings)
            # out1=self.decoder(x,src_encodings)
            out1=self.out(out1)
            # F.softmax(self.production_readout(query_vectors), dim=-1)
            apply_rule_log_prob = F.log_softmax(self.production_readout(out1[:,-1,:].squeeze(dim=1)), dim=-1)
            # print (apply_rule_log_prob.shape)
            # print ('apply_rule_log_prob shape',apply_rule_log_prob.shape)
            gen_from_vocab_prob = F.softmax(self.tgt_token_readout(out1[:,-1,:].squeeze(dim=1)), dim=-1)
            if args.no_copy:
                #TODO: to log or not?
                print ('no copy is true')
                primitive_prob = gen_from_vocab_prob
            else:
                # print (out1[:,-1,:].shape)
                primitive_copy_prob = self.src_pointer_net(src_encodings, None, out1[:,-1,:].unsqueeze(dim=0)).squeeze(0)
                primitive_predictor_prob = F.softmax(self.primitive_predictor(out1[:,-1,:]), dim=-1)
                primitive_prob = primitive_predictor_prob[:, 0].unsqueeze(1) * gen_from_vocab_prob
                # print (primitive_predictor_prob[:, 0].unsqueeze(1),'aha')


            

            gentoken_prev_hyp_ids = []
            gentoken_new_hyp_scores = []
            gentoken_new_hyp_unks = []
            applyrule_new_hyp_scores = []
            applyrule_new_hyp_prod_ids = []
            # applyrule_prev_hyp_ids = []

            action_types = self.transition_system.get_valid_continuation_types(hypothesis)
            # print (action_types, "these are the action types")
            for action_type in action_types:
                if action_type == ApplyRuleAction: 
                    
                    productions = self.transition_system.get_valid_continuating_productions(hypothesis)
                    # print (productions,'productions: ')
                    for production in productions:
                        prod_id = self.grammar.prod2id[production]
                        #apply_rule_log_prob is of size hyp_num,x_len(out_len) fed into the transformers, grammar_size
                        # print ('apply_rule_shape',apply_rule_log_prob.shape)
                        # print (apply_rule_log_prob.shape)
                        prod_score = apply_rule_log_prob[0, prod_id].data.item()
                        new_hyp_score = prod_score

                        applyrule_new_hyp_scores.append(new_hyp_score)
                        applyrule_new_hyp_prod_ids.append(prod_id)
                        # applyrule_prev_hyp_ids.append(hyp_id)
                elif action_type == ReduceAction:
                    assert apply_rule_log_prob.shape[-1]==len(self.grammar)+1
                    action_score = apply_rule_log_prob[0, -1].data.item()
                    new_hyp_score = action_score

                    applyrule_new_hyp_scores.append(new_hyp_score)
                    applyrule_new_hyp_prod_ids.append(len(self.grammar))
                    # applyrule_prev_hyp_ids.append(hyp_id)
                else:
                    gentoken_prev_hyp_ids.append(0)
                    hyp_copy_info = dict()  # of (token_pos, copy_prob)
                    hyp_unk_copy_info = []

                    if args.no_copy is False:
                        for token, token_pos_list in aggregated_primitive_tokens.items():
                            sum_copy_prob = torch.gather(primitive_copy_prob[0], 0, Variable(T.LongTensor(token_pos_list))).sum()
                            gated_copy_prob = primitive_predictor_prob[0, 1] * sum_copy_prob

                            if token in primitive_vocab:
                                token_id = primitive_vocab[token]
                                primitive_prob[0, token_id] = primitive_prob[0, token_id] + gated_copy_prob

                                hyp_copy_info[token] = (token_pos_list, gated_copy_prob.data.item())
                            else:
                                hyp_unk_copy_info.append({'token': token, 'token_pos_list': token_pos_list,
                                                            'copy_prob': gated_copy_prob.data.item()})

                    if args.no_copy is False and len(hyp_unk_copy_info) > 0:
                        unk_i = np.array([x['copy_prob'] for x in hyp_unk_copy_info]).argmax()
                        token = hyp_unk_copy_info[unk_i]['token']
                        primitive_prob[0, primitive_vocab.unk_id] = hyp_unk_copy_info[unk_i]['copy_prob']
                        gentoken_new_hyp_unks.append(token)

                        hyp_copy_info[token] = (hyp_unk_copy_info[unk_i]['token_pos_list'], hyp_unk_copy_info[unk_i]['copy_prob'])



                    # token_id=gen_from_vocab_prob.argmax(dim=-1)
                    # # if token_id==primitive_vocab.unk_id:
                    # #     print ('it is the unknown id')
                    # new_hyp_score=gen_from_vocab_prob[0,token_id].data.item()
                    # gentoken_new_hyp_ids.append(token_id)
                    # gentoken_new_hyp_scores.append(new_hyp_score)
                    # # print ('we haven\'t implemented it yet. Gentoken reached')
                 
            new_hyp_scores = None
            if applyrule_new_hyp_scores:
                new_hyp_scores = Variable(self.new_tensor(applyrule_new_hyp_scores))
            if gentoken_prev_hyp_ids:
                primitive_log_prob = torch.log(primitive_prob)
                if new_hyp_scores is None:new_hyp_scores = primitive_log_prob.view(-1)
                else: new_hyp_scores = torch.cat([new_hyp_scores,primitive_log_prob.view(-1)],dim=-1)
                # if new_hyp_scores is None: new_hyp_scores = Variable(self.new_tensor(gentoken_new_hyp_scores))
                # else: new_hyp_scores = torch.cat([new_hyp_scores, Variable(self.new_tensor(gentoken_new_hyp_scores))],dim=-1)
                # print ("gentoken reached again")
            
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores,
                                                             k=min(1,new_hyp_scores.shape[0]))
            
            # action_info = ActionInfo()
            # print (top_new_hyp_pos,type(top_new_hyp_pos))
            top_new_hyp_scores=top_new_hyp_scores[-1]
            top_new_hyp_pos=top_new_hyp_pos[-1]
            action_info = ActionInfo()
            if top_new_hyp_pos.item() < len(applyrule_new_hyp_scores):
            
                prod_id = applyrule_new_hyp_prod_ids[top_new_hyp_pos.data.cpu()]
                # ApplyRule action
                if prod_id < len(self.grammar):
                    # print ('does these thing match?')
                    # print (prod_id)
                    production = self.grammar.id2prod[prod_id]
                    # print (production)
                    action = ApplyRuleAction(production)
                # Reduce action
                else:
                    # print ('does these thing match?')
                    # print (prod_id)
                    # print (len(self.grammar)+1)
                    action = ReduceAction()
            else:
                token_id = (top_new_hyp_pos - len(applyrule_new_hyp_scores)) % primitive_prob.size(-1)
                k = (top_new_hyp_pos - len(applyrule_new_hyp_scores)) // primitive_prob.size(1)
                # token_id = gentoken_new_hyp_ids[top_new_hyp_pos.data.cpu()-len(applyrule_new_hyp_prod_ids)]
                # print (primitive_vocab.unk_id)
                if token_id == primitive_vocab.unk_id:
                    if gentoken_new_hyp_unks:
                        token = gentoken_new_hyp_unks[k]
                    else:
                        token = primitive_vocab.id2word[primitive_vocab.unk_id]
                else:
                    token = primitive_vocab.id2word[token_id.item()]

                action = GenTokenAction(token)
                if token in aggregated_primitive_tokens:
                        action_info.copy_from_src = True
                        action_info.src_token_position = aggregated_primitive_tokens[token]


            # action_info=ActionInfo()
            action_info.action = action
            action_info.t = a
            if a > 0:
                action_info.parent_t = hypothesis.frontier_node.created_time
                action_info.frontier_prod = hypothesis.frontier_node.production
                action_info.frontier_field = hypothesis.frontier_field.field
            new_hyp = hypothesis.clone_and_apply_action_info(action_info)

            
            if new_hyp.completed:
                completed_hypotheses.append(new_hyp)
                
                break
            else:

                hypothesis=new_hyp

                a+=1
                if a==200:
                    completed_hypotheses.append(hypothesis)
                    print ('too manyyyy')
                    # print(completed_hypotheses[0].actions)
                    # a+=1
                    return completed_hypotheses

        return completed_hypotheses

    

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'transition_system': self.transition_system,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)
 
    @classmethod
    def load(cls, model_path, cuda=False):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        transition_system = params['transition_system']
        saved_args = params['args']
        # update saved args
        update_args(saved_args, init_arg_parser())
        saved_state = params['state_dict']
        saved_args.cuda = cuda

        parser = cls(saved_args, vocab, transition_system)

        parser.load_state_dict(saved_state)

        if cuda: parser = parser.cuda()
        parser.eval()

        return parser



class PositionalEncoding(nn.Module):
    "Encode the position with a sinusoid."
    def __init__(self, d):
        super().__init__()
        self.register_buffer('freq', 1 / (10000 ** (torch.arange(0., d, 2.)/d)))

    def forward(self, pos):
        inp = torch.ger(pos, self.freq)
        enc = torch.cat([inp.sin(), inp.cos()], dim=-1)
        return enc

class TransformerEmbedding(nn.Module):
    "Embedding + positional encoding + dropout"
    def __init__(self, vocab_sz, emb_sz, inp_p=0.):
        super().__init__()
        self.emb_sz = emb_sz
        self.embed = embedding(vocab_sz, emb_sz)
        self.pos_enc = PositionalEncoding(emb_sz)
        self.drop = nn.Dropout(inp_p)

    def forward(self, inp):
        pos = torch.arange(0, inp.size(1), device=inp.device).float()
        return self.drop(self.embed(inp) * math.sqrt(self.emb_sz) + self.pos_enc(pos))

def feed_forward(d_model, d_ff, ff_p=0., double_drop=True):
    layers = [nn.Linear(d_model, d_ff), nn.ReLU()]
    if double_drop: layers.append(nn.Dropout(ff_p))
    return SequentialEx(*layers, nn.Linear(d_ff, d_model), nn.Dropout(ff_p), MergeLayer(), nn.LayerNorm(d_model))

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_head=None, p=0., bias=True, scale=True):
        super().__init__()
        d_head = ifnone(d_head, d_model//n_heads)
        self.n_heads,self.d_head,self.scale = n_heads,d_head,scale
        self.q_wgt,self.k_wgt,self.v_wgt = [nn.Linear(
            d_model, n_heads * d_head, bias=bias) for o in range(3)]
        self.out = nn.Linear(n_heads * d_head, d_model, bias=bias)
        self.drop_att,self.drop_res = nn.Dropout(p),nn.Dropout(p)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, q, kv, mask=None):
        return self.ln(q + self.drop_res(self.out(self._apply_attention(q, kv, mask=mask))))

    def create_attn_mat(self, x, layer, bs):
        return layer(x).view(bs, x.size(1), self.n_heads, self.d_head
                            ).permute(0, 2, 1, 3)

    def _apply_attention(self, q, kv, mask=None):
        bs,seq_len = q.size(0),q.size(1)
        wq,wk,wv = map(lambda o: self.create_attn_mat(*o,bs),
                       zip((q,kv,kv),(self.q_wgt,self.k_wgt,self.v_wgt)))
        attn_score = wq @ wk.transpose(2,3)
        if self.scale: attn_score /= math.sqrt(self.d_head)
        if mask is not None:
            attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=-1))
        attn_vec = attn_prob @ wv
        return attn_vec.permute(0, 2, 1, 3).contiguous().view(bs, seq_len, -1)

def get_output_mask(inp, pad_idx=0):
    return torch.triu(inp.new_ones(inp.size(1),inp.size(1)), diagonal=1)[None,None].bool()





class EncoderBlock(nn.Module):
    "Encoder block of a Transformer model."
    
    def __init__(self, n_heads, d_model, d_head, d_inner, p=0., bias=True, scale=True, double_drop=True):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.ff  = feed_forward(d_model, d_inner, ff_p=p, double_drop=double_drop)

    def forward(self, x, mask=None): return self.ff(self.mha(x, x, mask=mask))

class DecoderBlock(nn.Module):
    "Decoder block of a Transformer model."
    
    def __init__(self, n_heads, d_model, d_head, d_inner, p=0., bias=True, scale=True, double_drop=True):
        super().__init__()
        self.mha1 = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.mha2 = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.ff   = feed_forward(d_model, d_inner, ff_p=p, double_drop=double_drop)

    def forward(self, x, enc, mask_out=None): return self.ff(self.mha2(self.mha1(x, x, mask_out), enc))


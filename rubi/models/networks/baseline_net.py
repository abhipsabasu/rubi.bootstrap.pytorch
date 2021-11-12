from copy import deepcopy
import itertools
import os
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
import block
from block.models.networks.vqa_net import factory_text_enc
from block.models.networks.mlp import MLP

from .utils import mask_softmax

torch.set_printoptions(profile="full")


class BaselineNet(nn.Module):

    def __init__(self,
                 txt_enc={},
                 self_q_att=False,
                 agg={},
                 classif={},
                 wid_to_word={},
                 word_to_wid={},
                 aid_to_ans=[],
                 ans_to_aid={},
                 fusion={},
                 residual=False,
                 fusion_attn={}
                 ):
        super().__init__()
        self.self_q_att = self_q_att
        self.agg = agg
        assert self.agg['type'] in ['max', 'mean']
        self.classif = classif
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        self.fusion = fusion
        self.fusion_attn = fusion_attn
        self.residual = residual

        # Modules
        #self.Wt = nn.Linear(620, 620)
        # self.W = nn.Linear(2048, 2048)
        # self.wa = nn.Linear(2048, 1)
        self.Wq = nn.Linear(620, 512)
        self.txt_enc = self.get_text_enc(self.wid_to_word, txt_enc)
        if self.self_q_att:
            self.q_att_linear0 = nn.Linear(2048, 512)
            self.q_att_linear1 = nn.Linear(512, 2)
        #    self.q_att_linear2 = nn.Linear(620, 2048)
        #     self.q_att_linear3 = nn.Linear(100, 2048)
        self.rnn = nn.GRU(2048, 1024, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fusion_module = block.factory_fusion(self.fusion)
        self.fusion_attn_module = block.factory_fusion(self.fusion_attn)
        self.dropout = nn.Dropout(0.2)
        if self.classif['mlp']['dimensions'][-1] != len(self.aid_to_ans):
            Logger()(f"Warning, the classif_mm output dimension ({self.classif['mlp']['dimensions'][-1]})"
                     f"doesn't match the number of answers ({len(self.aid_to_ans)}). Modifying the output dimension.")
            self.classif['mlp']['dimensions'][-1] = len(self.aid_to_ans)

        self.classif_module = MLP(**self.classif['mlp'])

        Logger().log_value('nparams',
                           sum(p.numel() for p in self.parameters() if p.requires_grad),
                           should_print=True)

        Logger().log_value('nparams_txt_enc',
                           self.get_nparams_txt_enc(),
                           should_print=True)

    def get_text_enc(self, vocab_words, options):
        """
        returns the text encoding network.
        """
        return factory_text_enc(self.wid_to_word, options)

    def get_nparams_txt_enc(self):
        params = [p.numel() for p in self.txt_enc.parameters() if p.requires_grad]
        # if self.self_q_att:
        #     params += [p.numel() for p in self.q_att_linear0.parameters() if p.requires_grad]
        #     params += [p.numel() for p in self.q_att_linear1.parameters() if p.requires_grad]
        return sum(params)

    def process_fusion(self, q, mm):
        bsize = mm.shape[0]
        n_regions = mm.shape[1]

        mm = mm.contiguous().view(bsize * n_regions, -1)
        mm = self.fusion_module([q, mm])
        mm = mm.view(bsize, n_regions, -1)
        return mm

    def process_fusion_attn(self, q, mm):
        bsize = mm.shape[0]
        n_regions = mm.shape[1]

        mm = mm.contiguous().view(bsize * n_regions, -1)
        mm = self.fusion_attn_module([q, mm])
        mm = mm.view(bsize, n_regions, -1)
        return mm

    def forward(self, batch):
        v = batch['visual']  # Shape: Batch*36*2048
        q = batch['question']
        l = batch['lengths'].data
        cls_id = batch['cls_wid']
        c = batch['norm_coord']
        nb_regions = batch.get('nb_regions')
        bsize = v.shape[0]
        n_regions = v.shape[1]
        v = F.normalize(v, dim=-1)
        out = {}

        q = self.process_question(q, l, v, cls_id)  # Shape: Batch*4800
        out['q_emb'] = q
        q_expand = q[:, None, :].expand(bsize, n_regions, q.shape[1])  # Shape: Batch*36*4800
        q_expand = q_expand.contiguous().view(bsize * n_regions, -1)

        mm = self.process_fusion(q_expand, v, )  # Shape: Batch*36*2048
        if self.residual:
            mm = v + mm

        if self.agg['type'] == 'max':
            mm, mm_argmax = torch.max(mm, 1)
        elif self.agg['type'] == 'mean':
            mm = mm.mean(1)

        out['mm'] = mm
        out['mm_argmax'] = mm_argmax

        logits = self.classif_module(mm)
        out['logits'] = logits
        return out

    def process_question(self, q, l, v, cls_id, txt_enc=None, q_att_linear0=None, q_att_linear1=None):
        if txt_enc is None:
            txt_enc = self.txt_enc
        if q_att_linear0 is None:
            q_att_linear0 = self.q_att_linear0
        if q_att_linear1 is None:
            q_att_linear1 = self.q_att_linear1

        q_emb = txt_enc.embedding(q)  # Batch*Length*620
        #q_emb = F.normalize(q_emb, dim=-1)
        cls_id = cls_id.long()
        cls_id_exp = cls_id.contiguous().view(cls_id.shape[0] * cls_id.shape[1], -1)
        # print(cls_id_exp[0])
        cls_emb = txt_enc.embedding(cls_id_exp)
        # print(cls_emb)
        cls_emb = cls_emb.view(cls_id.shape[0], cls_id.shape[1], 2, -1)
        # print(cls_emb[0, 0])
        cls_emb = cls_emb.sum(2)  # Batch*36*620
        attn_scores = torch.bmm(q_emb, torch.transpose(cls_emb, 1, 2))
        attn_scores = torch.softmax(attn_scores, dim=2)
        attn_out = torch.bmm(attn_scores, v)
        #attn_out = self.dropout(attn_out)
        # print(q_emb[0, 2])

        # #q1 = self.q_att_linear2(q_emb)
        # #q1 = self.q_att_linear3(q1)  # Shape: batch*length*2048
        # cls_unsqueeze = torch.unsqueeze(cls_emb, dim=1)
        # cls_expand = cls_unsqueeze.expand(-1, q.shape[1], -1, -1)
        # q_unsqueeze = torch.unsqueeze(q_emb, dim=2)
        # q_expand = q_unsqueeze.expand(-1, -1, v.shape[1], -1)
        # attn_scores = torch.multiply(cls_expand, q_expand)
        # #attn_scores = torch.transpose(attn_scores, 2, 3)
        # attn_scores = self.W(attn_scores)
        # attn_scores = self.wa(attn_scores)
        # attn_scores = torch.softmax(attn_scores, dim=2)
        # cls_trans = torch.transpose(cls_emb, 1, 2)
        #
        # attn_scores = torch.bmm(q_emb, cls_trans)
        # attn_scores = torch.softmax(attn_scores, dim=2)  # Shape: batch*length*36
        # #print(attn_scores[0, 5], "AYABA")'''
        # v_unsqueeze = torch.unsqueeze(v, dim=1)
        # v_expand = v_unsqueeze.expand(-1, q.shape[1], -1, -1)
        # q_unsqueeze = torch.unsqueeze(q_emb, dim=2)
        # q_expand = q_unsqueeze.expand(-1, -1, v.shape[1], -1)
        # attn_scores = torch.multiply(v_expand, q_expand)
        # attn_scores = self.W(attn_scores)
        # attn_scores = self.wa(attn_scores)
        # attn_scores = torch.softmax(attn_scores, dim=2)
        # #attn_scores = attn_scores.unsqueeze(-1)
        # attn_scores = attn_scores.expand_as(v_expand)
        # attn_out = attn_scores * v_expand
        # attn_out = attn_out.sum(2) #Shape: batch*length*2048
        # #attn_out = attn_out.contiguous().view(q.shape[0] * q.shape[1], -1)
        #q_emb = self.Wt(q_emb)
        #q_emb = F.relu(q_emb)
        q_emb = self.Wq(q_emb)
        #q_emb = self.dropout(q_emb)
        q_emb_attn = q_emb.contiguous().view(q.shape[0] * q.shape[1], -1)
        mm = self.process_fusion_attn(q_emb_attn, attn_out)
        q, hidden_states = self.rnn(mm)  # Shape : batch*length*2400
        # print(q.size(), torch.cat((hidden_states[-1], hidden_states[-2]), dim=1).size())
        # return (torch.cat((hidden_states[-1], hidden_states[-2]), dim=1))
        if self.self_q_att:
            q_att = q_att_linear0(q)  # Shape: batch*length*512
            q_att = F.relu(q_att)
            q_att = q_att_linear1(q_att)  # Shape: batch*length*2
            q_att = mask_softmax(q_att, l)  # Shape: batch*length*2 - Obtain masked_softmax probabilities.
            # self.q_att_coeffs = q_att
            if q_att.size(2) > 1:
                q_atts = torch.unbind(q_att, dim=2)  # 2 length tuple: Individual Shape: batch * length
                q_outs = []
                for q_att in q_atts:
                    q_att = q_att.unsqueeze(2)  # Shape: Batch*length. Probability values summing to 1
                    q_att = q_att.expand_as(q)  # SHape: Batch*length*2400
                    q_out = q_att * q
                    q_out = q_out.sum(1)  # Shape: Batch*2400
                    q_outs.append(q_out)
                q = torch.cat(q_outs, dim=1)
            else:
                q_att = q_att.expand_as(q)
                q = q_att * q
                q = q.sum(1)
        else:
            # l contains the number of words for each question
            # in case of multi-gpus it must be a Tensor
            # thus we convert it into a list during the forward pass
            l = list(l.data[:, 0])
            q = txt_enc._select_last(q, l)
        #q = self.dropout(q)
        return q

    def process_answers(self, out, key=''):
        batch_size = out[f'logits{key}'].shape[0]
        _, pred = out[f'logits{key}'].data.max(1)
        pred.squeeze_()
        if batch_size != 1:
            out[f'answers{key}'] = [self.aid_to_ans[pred[i].item()] for i in range(batch_size)]
            out[f'answer_ids{key}'] = [pred[i].item() for i in range(batch_size)]
        else:
            out[f'answers{key}'] = [self.aid_to_ans[pred.item()]]
            out[f'answer_ids{key}'] = [pred.item()]
        return out

import torch.nn as nn
import os
import torch
import torch.nn.functional as F
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from block.models.criterions.vqa_cross_entropy import VQACrossEntropyLoss
import numpy as np


class ContrastiveCriterion(nn.Module):

    def __init__(self):
        super().__init__()

        Logger()(f'ContrastiveCriterion')
        self.cross_entropy = VQACrossEntropyLoss()
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, net_out, batch):
        #print(net_out)
        top_ans_emb = net_out[f'answer_ids']
        bsize = net_out['q_emb'].shape[0]
        best = net_out['ans_embedding'][top_ans_emb]  # torch.zeros(bsize, list(net_out['ans_embedding'].values())[0].shape[-1]).to('cuda:0')
        # best = torch.cat(net_out['ans_embedding'][i] for i in top_ans_emb).to('cuda:0')
        positive_dist = self.cos(net_out['mm_proj'], best)  # shape b,k;b,k-> b
        all_ans_embs = net_out['ans_embedding']  # torch.stack(list(net_out['ans_embedding'].values()))
        gen_embs = net_out['mm_proj'].unsqueeze(1)
        gen_embs = gen_embs.expand(-1, all_ans_embs.shape[0], -1)
        all_ans_embs = all_ans_embs.unsqueeze(0)
        all_ans_embs = all_ans_embs.expand(gen_embs.shape[0], -1, -1)
        d_logit = self.cos(gen_embs, all_ans_embs)
        num = torch.exp(positive_dist).squeeze(-1)
        den = torch.exp(d_logit).sum(-1)
        loss_nce = -1 * torch.log(num / den)
        loss_nce = loss_nce.mean()
        out = self.cross_entropy(net_out, batch)
        #loss_nce=0
        out['loss'] = loss_nce + out['loss']
        return out

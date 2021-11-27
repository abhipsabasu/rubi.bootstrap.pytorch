import torch.nn as nn
import torch
import torch.nn.functional as F
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options

class RUBiCriterion(nn.Module):

    def __init__(self, question_loss_weight=1.0):
        super().__init__()

        Logger()(f'RUBiCriterion, with question_loss_weight = ({question_loss_weight})')

        self.question_loss_weight = question_loss_weight
        self.fusion_loss = nn.CrossEntropyLoss()
        self.question_loss = nn.CrossEntropyLoss()
        self.cos = nn.CosineSimilarity(dim=-1)
     
    def forward(self, net_out, batch):
        out = {}
        class_id = batch['class_id'].squeeze(1)
        # top_ans_emb = net_out[f'answer_ids']
        best = net_out['ans_embedding'][class_id]
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
        obj_dist = self.cos(net_out['v_max'], net_out['mm'])
        obj_loss = 1 - obj_dist
        obj_loss = obj_loss.mean()
        # logits = net_out['logits']
        logits_q = net_out['logits_q']
        logits_rubi = net_out['logits_rubi']

        fusion_loss = (self.fusion_loss(logits_rubi, class_id) + obj_loss + loss_nce) / 3.0
        question_loss = self.question_loss(logits_q, class_id)
        loss = fusion_loss + self.question_loss_weight * question_loss

        out['loss'] = loss
        out['loss_mm_q'] = fusion_loss
        out['loss_q'] = question_loss
        return out

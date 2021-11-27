import sys
import copy
import torch
import torch.nn as nn
import os
import json
from bootstrap.lib.options import Options
from bootstrap.models.networks.data_parallel import DataParallel
from block.models.networks.vqa_net import VQANet as AttentionNet
from bootstrap.lib.logger import Logger
from block.models.networks.vqa_net import factory_text_enc
from block.datasets.vqa_utils import tokenize
from .baseline_net import BaselineNet
from .rubi import RUBiNet


def get_text_enc(vocab_words, options):
    """
    returns the text encoding network.
    """
    return factory_text_enc(vocab_words, options)


def factory(engine):
    mode = list(engine.dataset.keys())[0]
    dataset = engine.dataset[mode]
    opt = Options()['model.network']
    txt_enc = opt['txt_enc']
    awid_to_ans = torch.load(
        os.path.join('data', 'vqa', 'vqacp2', 'processed', 'nans,3000_minwcount,0_nlp,mcb_proc_split,train',
                     'awid_to_ans.pth'))
    ans_to_awid = torch.load(
        os.path.join('data', 'vqa', 'vqacp2', 'processed', 'nans,3000_minwcount,0_nlp,mcb_proc_split,train',
                     'ans_to_awid.pth'))
    txt_enc_ans = get_text_enc(awid_to_ans, txt_enc).to('cuda:0')
    answers = {i: tokenize(a) for (a, i) in dataset.ans_to_aid.items()}
    answers = {i: torch.tensor([ans_to_awid[i] for i in tokens]).to('cuda:0') for i, tokens in answers.items()}
    answers = {i: txt_enc_ans.embedding(tokens.to('cuda:0')) for i, tokens in answers.items()}
    ans_emb = {i: torch.sum(embeddings, dim=0) for i, embeddings in answers.items()}
    if opt['name'] == 'baseline':

        net = BaselineNet(
            txt_enc=opt['txt_enc'],
            self_q_att=opt['self_q_att'],
            agg=opt['agg'],
            classif=opt['classif'],
            wid_to_word=dataset.wid_to_word,
            word_to_wid=dataset.word_to_wid,
            aid_to_ans=dataset.aid_to_ans,
            ans_to_aid=dataset.ans_to_aid,
            fusion=opt['fusion'],
            fusion_attn=opt['fusion_attn'],
            residual=opt['residual'],
            ans_emb=ans_emb
        )

    elif opt['name'] == 'rubi':
        orig_net = BaselineNet(
            txt_enc=opt['txt_enc'],
            self_q_att=opt['self_q_att'],
            agg=opt['agg'],
            classif=opt['classif'],
            wid_to_word=dataset.wid_to_word,
            word_to_wid=dataset.word_to_wid,
            aid_to_ans=dataset.aid_to_ans,
            ans_to_aid=dataset.ans_to_aid,
            fusion=opt['fusion'],
            fusion_attn=opt['fusion_attn'],
            residual=opt['residual'],
            ans_emb = ans_emb
        )
        net = RUBiNet(
            model=orig_net,
            output_size=len(dataset.aid_to_ans),
            classif=opt['rubi_params']['mlp_q']
        )
    else:
        raise ValueError(opt['name'])

    if Options()['misc.cuda'] and torch.cuda.device_count() > 1:
        net = DataParallel(net)

    return net


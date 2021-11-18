import torch
import os
from block.datasets.vqa_utils import tokenize


def convert_aid_word_id(ans_file, dir_path):
    ans = torch.load(ans_file)
    ans_words = []
    for a in ans:
        a = tokenize(a)
        ans_words = ans_words + a
    ans_words = list(set(ans_words))
    awid = {(i+1):ans_words[i] for i in range(len(ans_words))}
    awid_path = os.path.join(dir_path, 'awid_to_ans.pth')
    ans_path = os.path.join(dir_path, 'ans_to_awid.pth')
    ans_to_awid = {v:k for (k,v) in awid.items()}
    torch.save(awid, awid_path)
    torch.save(ans_to_awid, ans_path)

if __name__=='__main__':
    dir_path = os.path.join('data', 'vqa', 'vqacp2', 'processed', 'nans,3000_minwcount,0_nlp,mcb_proc_split,train')
    ans_file = os.path.join(dir_path, 'ans_to_aid.pth')
    convert_aid_word_id(ans_file, dir_path)
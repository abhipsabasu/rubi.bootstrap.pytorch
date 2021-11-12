import torch
import os

vqacp2_path = '../../data/vqa/vqacp2'
classes = ['__background__']
f = open(os.path.join(vqacp2_path, 'vgenome-features-object-vocab.txt'))
for line in f:
    classes.append(line.split(',')[0].strip())
f.close()
word_to_wid_path = os.path.join(vqacp2_path, 'processed', 'nans,3000_minwcount,0_nlp,mcb_proc_split,train', 'word_to_wid.pth')
wid_to_word_path = os.path.join(vqacp2_path, 'processed', 'nans,3000_minwcount,0_nlp,mcb_proc_split,train', 'wid_to_word.pth')
word_to_wid = torch.load(word_to_wid_path)
wid_to_word = torch.load(wid_to_word_path)
max_value = max(word_to_wid.values())
max_len = max([len(c.split()) for c in classes])
for c in classes:
    c_list = c.split()
    for c_ in c_list:
        if c_ not in word_to_wid:
            word_to_wid[c_] = max_value + 1
            wid_to_word[max_value+1] = c_
            max_value = max_value + 1
object_path = '../../data/vqa/coco/extract_rcnn/2018-04-27_bottom-up-attention_fixed_36'
torch.save(word_to_wid, word_to_wid_path)
torch.save(wid_to_word, wid_to_word_path)
for file in os.listdir(object_path):
    item = torch.load(os.path.join(object_path, file))
    cls = item['cls']
    cls = cls.tolist()
    num_obj = len(cls)
    item['cls_wid'] = torch.zeros((num_obj, max_len))
    for j in range(len(cls)):
        c = cls[j]
        words = classes[c].split()
        for i in range(len(words)):
            item['cls_wid'][j][i] = word_to_wid[words[i]]
    torch.save(item, os.path.join(object_path, file))
    print(file)
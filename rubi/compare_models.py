import json
import os
from sklearn.metrics import jaccard_score


def compare_models(model1, model2, dir):
    model1_train_path = os.path.join(dir, model1, 'results', 'train', 'epoch,21', 'OpenEnded_mscoco_train2014_model_results.json')
    model2_train_path = os.path.join(dir, model2, 'results', 'train', 'epoch,21', 'OpenEnded_mscoco_train2014_model_results.json')
    model1_eval_path = os.path.join(dir, model1, 'results', 'val', 'epoch,21', 'OpenEnded_mscoco_val2014_model_results.json')
    model2_eval_path = os.path.join(dir, model2, 'results', 'val', 'epoch,21', 'OpenEnded_mscoco_val2014_model_results.json')
    with open(model1_train_path) as f:
        model1_train_json = json.load(f)
    with open(model2_train_path) as f:
        model2_train_json = json.load(f)
    with open(model1_eval_path) as f:
        model1_eval_json = json.load(f)
    with open(model2_eval_path) as f:
        model2_eval_json = json.load(f)
    #print(set([dic['question_id'] for dic in model1_eval_json if dic['answer']==dic['gt']]))
    model1_train_set = set([dic['question_id'] for dic in model1_train_json if dic['answer'] == dic['gt']])
    model2_train_set = set([dic['question_id'] for dic in model2_train_json if dic['answer'] == dic['gt']])
    model1_eval_set = set([dic['question_id'] for dic in model1_eval_json if dic['answer'] == dic['gt']])
    model2_eval_set = set([dic['question_id'] for dic in model2_eval_json if dic['answer'] == dic['gt']])
    train_jaccard = len(model1_train_set & model2_train_set) / len(model1_train_set | model2_train_set)
    test_jaccard = len(model1_eval_set & model2_eval_set) / len(model1_eval_set | model2_eval_set)
    percentage_train = len(model2_train_set-model1_train_set) / len(model2_train_set)
    percentage_test = len(model2_eval_set - model1_eval_set) / len(model2_eval_set)
    return train_jaccard, test_jaccard, percentage_train, percentage_test

if __name__=="__main__":
    input_dir = os.path.join('logs', 'vqacp2')
    train_score, test_score, per_train, per_test = compare_models('baseline', 'vgqe_hyp', input_dir)
    print('Training Jaccard Score', train_score)
    print('Eval Jaccard Score', test_score)
    print('Percentage of correct answers in model 2 not in model 1 (train)', per_train)
    print('Percentage of correct answers in model 2 not in model 1 (eval)', per_test)

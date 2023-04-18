#Eval summarization by ROUGE score.
#We use rouge_score to cal score: https://pypi.org/project/rouge-score/

import json
import re
import glob
from tqdm import tqdm
import argparse
import random
import os
from collections import defaultdict, Counter
import math
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from rouge_score import rouge_scorer


parser = argparse.ArgumentParser() 
parser.add_argument('--max_data_size', type=int, default=10000000000)
parser.add_argument('--data_path', type=str, default="out/cnndm_large")
parser.add_argument('--check_selfbleu', action="store_true")
args = parser.parse_args() 


with open(os.path.join(args.data_path, "generated_data.json"), encoding="utf-8")as f:
    data=json.load(f)

reference=[d["reference"] for d in data]
hypothesis=[d["hypothesis"] for d in data]


print("Use rouge_scorer")



scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
rouge1_precision=[]
rouge1_recall=[]
rouge1_fmeasure=[]
rouge2_fmeasure=[]
rougel_fmeasure=[]

for r,h in tqdm(zip(reference,hypothesis)):
    r=" ".join(word_tokenize(r))
    h=" ".join(word_tokenize(h))

    r="\n".join(sent_tokenize(r))
    h="\n".join(sent_tokenize(h))
    score = scorer.score(r, h)
    rouge1_precision.append(score["rouge1"].precision)
    rouge1_recall.append(score["rouge1"].recall)
    rouge1_fmeasure.append(score["rouge1"].fmeasure)
    rouge2_fmeasure.append(score["rouge2"].fmeasure)
    rougel_fmeasure.append(score["rougeLsum"].fmeasure)

print("precision", np.average(rouge1_precision))
print("recall", np.average(rouge1_recall))
print("fmeasure", np.average(rouge1_fmeasure))
print("rouge2_fmeasure", np.average(rouge2_fmeasure))
print("rougel_fmeasure", np.average(rougel_fmeasure))

print()



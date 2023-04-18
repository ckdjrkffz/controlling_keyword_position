#Evaluation keyword inclusion and position control.
#This codes can be used for multiple keyword setting.

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
from nltk import bleu_score
from nltk.tokenize import word_tokenize



parser = argparse.ArgumentParser() 
parser.add_argument('--data_size', type=int, default=10000000000)
parser.add_argument('--data_path', type=str, default="")
parser.add_argument('--check_selfbleu', action="store_true")
parser.add_argument('--lsep', type=int, default=5)
parser.add_argument('--psep', type=int, default=10)
parser.add_argument('--keyword_data_path', type=str, default="None")
args = parser.parse_args() 

mecab=MeCab.Tagger("-Ochasen")

print(args.data_path)

print("process dataset")

with open(os.path.join(args.data_path, "generated_data.json"), encoding="utf-8")as f:
    data=json.load(f)

if args.keyword_data_path!="None":
    with open(os.path.join(args.keyword_data_path, "generated_data.json"), encoding="utf-8")as f:
        keyword_data=json.load(f)
else:
    keyword_data=[None]*len(data)



reference_length_list=[]
hypothesis_length_list=[]
specified_length_list=[]

hypothesis_keyword_find_list=[]
hypothesis_keyword_position_list=[]
hypothesis_keyword_relative_position_list=[]
reference_keyword_find_list=[]
reference_keyword_position_list=[]
reference_keyword_relative_position_list=[]

specified_keyword_relative_position_list=[]

not_found_keyword_list=[]
found_keyword_list=[]
keyword_list=[]

length_diff_list = []
keyword_relative_position_diff_list = []


all_keyword_find_flag_list = []
all_keyword_position_correct_flag_list = []



for output_data, keyword_output_data in tqdm(zip(data[:args.data_size], keyword_data[:args.data_size])):

    if keyword_output_data == None:
        specified_length=output_data["reference_length"]
        specified_keyword=output_data["reference_keyword"]    
    else:
        specified_length=keyword_output_data["reference_length"]
        specified_keyword=keyword_output_data["reference_keyword"]    


    reference_text=output_data["reference"]
    reference_text_tokenized=word_tokenize(reference_text)
    hypothesis_text=output_data["hypothesis"]
    hypothesis_text_tokenized=word_tokenize(hypothesis_text)

    reference_length_list.append(len(reference_text_tokenized))
    hypothesis_length_list.append(len(hypothesis_text_tokenized))
    specified_length_list.append(specified_length)

    reference_length_min = int(len(reference_text_tokenized)/args.lsep)*args.lsep
    reference_length_max = reference_length_min + args.lsep - 0.0000001
    hypothesis_length = len(hypothesis_text_tokenized)

    length_diff_list.append(
            max(
                max(reference_length_min-hypothesis_length,0), 
                max(hypothesis_length-reference_length_max,0)
            )
        )

    all_keyword_find_flag=True
    all_keyword_position_correct_flag=True

    #if len(specified_keyword)<=1:
    #    continue

    for item in specified_keyword:
        #keyword_token, keyword_position=item["token"], int(item["position"])
        keyword_token, specified_keyword_position=item["token"], item["position"]

        #keyword
        keyword_list.append(" ".join(keyword_token))

        #hypothesis
        find_keyword_flag=0
        hypothesis_keyword_position=-1
        for i in range(len(hypothesis_text_tokenized)-len(keyword_token)+1):
            if keyword_token==hypothesis_text_tokenized[i:i+len(keyword_token)]:
                find_keyword_flag=True
                hypothesis_keyword_position=i
                break

        if find_keyword_flag==False:
            all_keyword_find_flag=False

        hypothesis_keyword_relative_position = \
            hypothesis_keyword_position/len(hypothesis_text_tokenized)*100 if hypothesis_keyword_position!=-1 else \
            -1

        
        #reference
        find_keyword_flag=0
        reference_keyword_position=-1
        for i in range(len(reference_text_tokenized)-len(keyword_token)+1):
            if keyword_token==reference_text_tokenized[i:i+len(keyword_token)]:
                find_keyword_flag=1
                reference_keyword_position=i
                break

        reference_keyword_relative_position = \
            reference_keyword_position/len(reference_text_tokenized)*100 if reference_keyword_position!=-1 else \
            -1

        #hypothesis and reference_diff
        reference_keyword_relative_position_min = \
            int(reference_keyword_relative_position/args.psep)*args.psep
        reference_keyword_relative_position_max = \
            reference_keyword_relative_position_min + args.psep - 0.0000001
        hypothesis_keyword_relative_position = hypothesis_keyword_relative_position

        keyword_position_diff = \
                -1 if hypothesis_keyword_relative_position==-1 else \
                max(
                    max(reference_keyword_relative_position_min-hypothesis_keyword_relative_position, 0), 
                    max(hypothesis_keyword_relative_position-reference_keyword_relative_position_max, 0)
                )
        
        if keyword_position_diff !=0:
            all_keyword_position_correct_flag=False

    all_keyword_find_flag_list.append(all_keyword_find_flag)
    all_keyword_position_correct_flag_list.append(all_keyword_position_correct_flag)



print("list size:", len(all_keyword_find_flag_list))

print("all keyword find", len([1 for flag in all_keyword_find_flag_list if flag==True])/len(all_keyword_find_flag_list))
print("all keyword position correct", len([1 for flag in all_keyword_position_correct_flag_list if flag==True])/len(all_keyword_position_correct_flag_list))


#Eval control tokens

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
from nltk.tokenize import word_tokenize



parser = argparse.ArgumentParser() 
parser.add_argument('--data_size', type=int, default=10000000000)
parser.add_argument('--data_path', type=str)
parser.add_argument('--check_selfbleu', action="store_true")
parser.add_argument('--lsep', type=int, default=5)
parser.add_argument('--psep', type=int, default=10)
parser.add_argument('--keyword_data_path', type=str, default="None")
args = parser.parse_args() 


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


    for item in specified_keyword:
        keyword_token, specified_keyword_position=item["token"], item["position"]

        keyword_list.append(" ".join(keyword_token))
        

        #hypothesis
        find_keyword_flag=0
        hypothesis_keyword_position=-1
        for i in range(len(hypothesis_text_tokenized)-len(keyword_token)+1):
            if keyword_token==hypothesis_text_tokenized[i:i+len(keyword_token)]:
                find_keyword_flag=1
                hypothesis_keyword_position=i
                break

        hypothesis_keyword_relative_position = \
            hypothesis_keyword_position/len(hypothesis_text_tokenized)*100 if hypothesis_keyword_position!=-1 else \
            -1

        hypothesis_keyword_find_list.append(find_keyword_flag)
        hypothesis_keyword_position_list.append(hypothesis_keyword_position)
        hypothesis_keyword_relative_position_list.append(hypothesis_keyword_relative_position)
        if find_keyword_flag==0:
            not_found_keyword_list.append(" ".join(keyword_token))
        else:
            found_keyword_list.append(" ".join(keyword_token))


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

        reference_keyword_find_list.append(find_keyword_flag)
        reference_keyword_position_list.append(reference_keyword_position)
        reference_keyword_relative_position_list.append(reference_keyword_relative_position)

        #specified
        specified_keyword_relative_position_list.append(\
            int(specified_keyword_position) if specified_keyword_position!="None" else -1)


        #hypothesis and reference_diff
        reference_keyword_relative_position_min = \
            int(reference_keyword_relative_position/args.psep)*args.psep
        reference_keyword_relative_position_max = \
            reference_keyword_relative_position_min + args.psep - 0.0000001
        hypothesis_keyword_relative_position = hypothesis_keyword_relative_position

        keyword_relative_position_diff_list.append(
                hypothesis_keyword_relative_position if hypothesis_keyword_relative_position==-1 else \
                max(
                    max(reference_keyword_relative_position_min-hypothesis_keyword_relative_position, 0), 
                    max(hypothesis_keyword_relative_position-reference_keyword_relative_position_max, 0)
                )
            )


# Show some samples. 
print("------------------sample-------------------------")
for idx, output_data in tqdm(enumerate(data[:0])):
    try:
        print("reference:", output_data["reference"])
        print("hypothesis:", output_data["hypothesis"])
        print("keyword:", output_data["reference_keyword"])
        print(hypothesis_keyword_find_list[idx])
        print("--End of sample-----")
    except:
        pass



print("------------------length control-------------------------")
print("r-h length diff", np.average([abs(r-h) for r,h in zip(reference_length_list, hypothesis_length_list)]))
print("r & h length average", np.average(reference_length_list), np.average(hypothesis_length_list))
for k,v in sorted(Counter([int(abs(r-h)/5)*5 for r,h in zip(reference_length_list, hypothesis_length_list)]).items(), key=lambda x:x[0]):
    print(k,v)


length_diff_list_counter=Counter(length_diff_list)
print("length diff 0: ", 
    round(np.sum([v for k,v in length_diff_list_counter.items() if k==0])/len(length_diff_list)*100, 2))
print("length diff 1-5: ", 
    round(np.sum([v for k,v in length_diff_list_counter.items() if 0<k<=5])/len(length_diff_list)*100, 2))
print("length diff 6-10: ", 
    round(np.sum([v for k,v in length_diff_list_counter.items() if 5<k<=10])/len(length_diff_list)*100, 2))
print("length diff 11-: ", 
    round(np.sum([v for k,v in length_diff_list_counter.items() if 10<k])/len(length_diff_list)*100, 2))
print()  


print("----------------- keyword inclusion-----------------------------")
print("r & h keyword find", np.average(reference_keyword_find_list), np.average(hypothesis_keyword_find_list))



print("------------------position control------------------------------")
print("r_pos & h_pos average", \
    np.average(reference_keyword_position_list), 
    np.average([pos for pos in hypothesis_keyword_position_list if pos!=-1]))
print("r-h pos diff", np.average(\
    [abs(r-h) for r,h in zip(reference_keyword_position_list, hypothesis_keyword_position_list) if h!=-1]))
print("r-h relative pos diff", np.average(\
    [abs(r-h) for r,h in zip(reference_keyword_relative_position_list, hypothesis_keyword_relative_position_list) \
        if h!=-1]))
print("s-h relative pos diff", np.average(\
    [abs(s-h) for s,h in zip(specified_keyword_relative_position_list, hypothesis_keyword_relative_position_list) \
        if s!=-1 and h!=-1]))
print("s-r relative pos diff", np.average(\
    [abs(s-r) for s,r in zip(specified_keyword_relative_position_list, reference_keyword_relative_position_list) \
        if s!=-1 and r!=-1]))


keyword_relative_position_diff_list_counter=Counter([int(p/10)*10 if p!=-1 else -1 for p in hypothesis_keyword_relative_position_list])
for k,v in sorted(keyword_relative_position_diff_list_counter.items(), key=lambda x:x[0]):
    print("position: {}\t{}\t{}".format(k,v,float(v/len(keyword_relative_position_diff_list))))


keyword_relative_position_diff_list_counter=Counter(keyword_relative_position_diff_list)
print("position diff -1: ", 
    round(np.sum([v for k,v in keyword_relative_position_diff_list_counter.items() if k==-1])/len(keyword_relative_position_diff_list)*100, 2))
print("position diff 0: ", 
    round(np.sum([v for k,v in keyword_relative_position_diff_list_counter.items() if k==0])/len(keyword_relative_position_diff_list)*100, 2))
print("position diff 0-10: ", 
    round(np.sum([v for k,v in keyword_relative_position_diff_list_counter.items() if 0<k<=10])/len(keyword_relative_position_diff_list)*100, 2))
print("position diff 10-20: ", 
    round(np.sum([v for k,v in keyword_relative_position_diff_list_counter.items() if 10<k<=20])/len(keyword_relative_position_diff_list)*100, 2))
print("position diff 20-: ", 
    round(np.sum([v for k,v in keyword_relative_position_diff_list_counter.items() if 20<k])/len(keyword_relative_position_diff_list)*100, 2))
print()  


print("----------Eval the keyword inclusion and position control for each target position----------------")

position_accuracy_list = []
for target_pos in range(0, 100, 10):
    keyword_relative_position_diff_list_sub = \
        [diff for diff, pos in \
            zip(keyword_relative_position_diff_list, reference_keyword_relative_position_list) \
                if target_pos<=pos<target_pos+10]

    keyword_relative_position_diff_list_counter=Counter(keyword_relative_position_diff_list_sub)
    position_accuracy_list.append([
        target_pos,
        len(keyword_relative_position_diff_list_sub),
        round(np.sum([v for k,v in keyword_relative_position_diff_list_counter.items() if k==0])/len(keyword_relative_position_diff_list_sub)*100, 2),
        round(np.sum([v for k,v in keyword_relative_position_diff_list_counter.items() if 0<k<=10])/len(keyword_relative_position_diff_list_sub)*100, 2),
        round(np.sum([v for k,v in keyword_relative_position_diff_list_counter.items() if 10<k])/len(keyword_relative_position_diff_list_sub)*100, 2),
        round(np.sum([v for k,v in keyword_relative_position_diff_list_counter.items() if k==-1])/len(keyword_relative_position_diff_list_sub)*100, 2),
    ])


print("Eval for each target position. We list: Target position, count, diff==0, 0<diff<=10, 10<diff, keyword not included")
for item in position_accuracy_list:
    print(" ".join([str(num) for num in item]))

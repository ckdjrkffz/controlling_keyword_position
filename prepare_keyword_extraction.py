#Extract control tokens from target texts.

import json
from tqdm import tqdm
import argparse
import random
import os
from collections import Counter
import itertools
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from glob import glob
from copy import deepcopy


parser = argparse.ArgumentParser() 
parser.add_argument('--data_size', type=int, default=10000000000)
parser.add_argument('--test_data_size', type=int, default=10000000000)
parser.add_argument('--data_path', type=str)
parser.add_argument('--use_diverse_position', action="store_true")
parser.add_argument('--num_keywords', type=int, default=1)
parser.add_argument('--use_diverse_keywords', action="store_true")
parser.add_argument('--use_all_keywords', action="store_true")
parser.add_argument('--not_all_stop', action="store_true")
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args() 



class KeywordExtractor:
    def __init__(self, tokenized_texts):
        #stop word append cand: 
        # https://gist.github.com/sebleier/554280  https://www.computerhope.com/jargon/s/specchar.htm ascii-code
        
        nltk_stop_words = stopwords.words('english')
        symbols=["'", '"', ':', ';', '.', ',', '-', '!', '?', "'s", "(", ")", "[", "]", "#", "%", "$", "@", "-", "+",
        "`", "~", "*", "|", "/", "\\", "<", ">", "''", '""', "``"]

        tokenized_words = [token.lower() for token in list(itertools.chain.from_iterable(tokenized_texts))]
        frequent_words = [k for k,v in sorted(Counter(tokenized_words).items(), key=lambda x:-x[1])][0:100]

        self.stop_words = set(nltk_stop_words + symbols + frequent_words)



    def extract(self, text, dataset_type):
        phrases=[]
        positions=[]
        relative_positions=[]

        keyword_size = \
            3 if dataset_type == "train" else \
            3

        text_tokenized=[(t,(t.lower() not in self.stop_words))  for t in text]
        length = len(text_tokenized)

        for span_length in range(1, keyword_size+1):
            for start in range(len(text_tokenized)-span_length+1):
                phrase=text_tokenized[start:start+span_length]
                if (args.not_all_stop and any([t[1] for t in phrase])) or \
                    (args.not_all_stop==False and phrase[0][1]==True):
                    phrases.append(" ".join([t[0] for t in phrase]))
                    positions.append(start)
                    relative_positions.append(start/length)

        if dataset_type =="test" and args.use_diverse_keywords==False and args.use_all_keywords==False:

            #shuffle
            random_positions = random.sample(list(range(len(phrases))), len(phrases))
            phrases = [phrases[pos] for pos in random_positions]
            positions = [positions[pos] for pos in random_positions]
            relative_positions = [relative_positions[pos] for pos in random_positions]

            #extract keyword (remove overlap)
            phrases_removed=[]
            positions_removed=[]
            relative_positions_removed=[]

            for phrase1, position1 ,relative_position1 in zip(phrases, positions, relative_positions):
                phrase_find_flag=False
                for phrase2 in phrases_removed:
                    if phrase1 in phrase2 or phrase2 in phrase1:
                        phrase_find_flag=True
                        break
                if phrase_find_flag==False:
                    phrases_removed.append(phrase1)
                    positions_removed.append(position1)
                    relative_positions_removed.append(relative_position1)
                
                if len(phrases_removed)==args.num_keywords:
                    break

            phrases = phrases_removed
            positions = positions_removed
            relative_positions = relative_positions_removed

        return phrases, positions, relative_positions, length


random.seed(args.seed)
path_list = [\
    os.path.join(args.data_path, "train.json"),
    os.path.join(args.data_path, "val.json"),
    os.path.join(args.data_path, "test.json"),
    ]

for path in path_list:
    print(f"Process {path}")
    with open(path)as f:
        data=json.load(f)

    dataset_type = \
        "train" if "train" in path else \
        "test"

    tokenized_texts = [word_tokenize(d["target"]) for d in tqdm(data)]   

    if dataset_type=="train":
        keyword_extractor=KeywordExtractor(tokenized_texts)

    if dataset_type=="test":
        tokenized_texts=tokenized_texts[0:args.test_data_size]
    processed_data=[]

    for i in tqdm(range(len(tokenized_texts))):
        keyword, position, relative_position, length = \
            keyword_extractor.extract(tokenized_texts[i], dataset_type = dataset_type)

        #for position diverse evaluation
        if dataset_type=="test" and args.use_diverse_position:
            for j in range(10):
                append_row = deepcopy(data[i])
                append_row["keyword"] = keyword
                append_row["position"] = position
                append_row["length"] = length
                append_row["relative_position"] = [j/10]
                processed_data.append(append_row)

        #for extract examples
        elif dataset_type=="test" and args.use_diverse_keywords:
            #100 keywords, 10 positions
            for _ in range(100): 
                num_keywords = random.choice([1, 1, 1, 1, 1, 2, 2, 2, 3, 3])
                random_positions = random.sample(list(range(len(keyword))), num_keywords)
                for keyword_position in range(10):
                    append_row = deepcopy(data[i])
                    append_row["keyword"] = [keyword[pos] for pos in random_positions]
                    append_row["position"] = [position[pos] for pos in random_positions]
                    append_row["length"] = length
                    if num_keywords == 1:
                        append_row["relative_position"] = [keyword_position/10 for pos in random_positions]
                    else:
                        append_row["relative_position"] = [int(random.random()*10)/10 for pos in random_positions]
                    processed_data.append(append_row)

        #normal
        else:
            if dataset_type=="test" and len(keyword)!=args.num_keywords and args.use_all_keywords==False:
                continue
            append_row = data[i]
            append_row["keyword"] = keyword
            append_row["position"] = position
            append_row["relative_position"] = relative_position
            append_row["length"] = length
            processed_data.append(append_row)


    print(f"Path:{path}, Data size:{len(processed_data)}")

    with open(path, "w", encoding="utf-8")as f:
        json.dump(processed_data, f, indent=4)


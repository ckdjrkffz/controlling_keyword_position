#cnndm preprocess

import json
import re
import glob
from tqdm import tqdm
import argparse
import random
import os
import itertools

parser = argparse.ArgumentParser() 
parser.add_argument('--max_data_size', type=int, default=10000000000)
parser.add_argument('--data_path', type=str)
args = parser.parse_args() 

dataset=[] 
random.seed(0)

os.makedirs(args.data_path, exist_ok=True)

for data_type in ["train", "val", "test"]:
    print("data process: ", data_type)
    source_path=os.path.join(args.data_path, data_type+".source")
    target_path=os.path.join(args.data_path, data_type+".target")
    output_path=os.path.join(args.data_path, data_type+".json")

    source_dataset=[]               
    with open(source_path, encoding="utf-8")as f:
        for l in f:
            source_dataset.append(l.strip())
    target_dataset=[]               
    with open(target_path, encoding="utf-8")as f:
        for l in f:
            target_dataset.append(l.strip())

    data=[]
    for source,target in zip(source_dataset, target_dataset):
        data.append({\
            "source": source,
            "target": target,
        })

    print(f"{data_type} data size: {len(data)}")

    
    with open(output_path, "w", encoding="utf-8")as f:
        json.dump(data, f, indent=4)

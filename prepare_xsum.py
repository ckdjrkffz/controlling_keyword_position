#XSum preprocess

import json
import re
import glob
from tqdm import tqdm
import argparse
import random
import os
import itertools
from datasets import load_dataset

parser = argparse.ArgumentParser() 
parser.add_argument('--max_data_size', type=int, default=10000000000)
parser.add_argument('--data_path', type=str)
args = parser.parse_args() 

dataset=[] 
random.seed(0)
os.makedirs(args.data_path, exist_ok=True)

dataset = load_dataset("xsum")

for data_type, dataset_id in [("train", "train"), ("val", "validation"), ("test", "test")]:
    print("Data process: ", data_type)
    output_path=os.path.join(args.data_path, data_type+".json")

    output_data= \
        [{"source": d["document"].replace("\n", " "),
        "target": d["summary"].replace("\n", " ")} \
            for d in dataset[dataset_id]]

    print(f"{data_type} data size: {len(output_data)}")

    with open(output_path, "w", encoding="utf-8")as f:
        json.dump(output_data, f, indent=4)

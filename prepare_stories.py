#stories preprocess

import json
import re
from glob import glob
from tqdm import tqdm
import argparse
import random
import os
import itertools
import pandas as pd

parser = argparse.ArgumentParser() 
parser.add_argument('--data_size', type=int, default=10000000000)
parser.add_argument('--data_path', type=str)
args = parser.parse_args() 

data=[] 
random.seed(0)

data=[]
columns=["sentence"+str(i) for i in range(5)]

for file in glob(os.path.join(args.data_path,"*.csv")):
    read_data=pd.read_csv(file).values
    for d in read_data:
        data.append(" ".join(d[2:7]))

data=[{"target":d} for d in data]
data=random.sample(data, len(data))
train_data=data[int(len(data)*0.0):int(len(data)*0.8)]
val_data=data[int(len(data)*0.8):int(len(data)*0.9)]
test_data=data[int(len(data)*0.9):int(len(data)*1.0)]

print("All data size: ",len(data))
print("Train data size: ",len(train_data))
print("Val data size: ",len(val_data))
print("Test data size: ",len(test_data))

with open(os.path.join(args.data_path, "train.json"), "w", encoding="utf-8")as f:
    json.dump(train_data, f, indent=4)
with open(os.path.join(args.data_path, "val.json"), "w", encoding="utf-8")as f:
    json.dump(val_data, f, indent=4)
with open(os.path.join(args.data_path, "test.json"), "w", encoding="utf-8")as f:
    json.dump(test_data, f, indent=4)



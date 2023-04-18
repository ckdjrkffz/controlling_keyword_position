# Train model & Generate texts code
# This code was created based on huggingface example codes (https://github.com/huggingface/transformers)


# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import logging
import math
import os
from dataclasses import dataclass, field
from glob import glob
import os
import pickle
import random
import time
from typing import Dict, List, Optional
import argparse
from datetime import datetime
from tqdm import tqdm
from collections import Counter, defaultdict
import itertools
import numpy as np
import re
import json
import io,sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from nltk.tokenize import word_tokenize

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
    AdamW,
    get_linear_schedule_with_warmup,
    T5Tokenizer,
    GPT2LMHeadModel,
    default_data_collator,
    BartModel,
    BartForConditionalGeneration,
    MBartForConditionalGeneration,
    T5ForConditionalGeneration,
    T5Config,
    MBartTokenizer,
    BartTokenizer,
    AutoModelForSeq2SeqLM,
    GPT2Tokenizer, 
    GPT2Model,
    AutoModelForCausalLM
)


logger = logging.getLogger(__name__)




class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self,
        args,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        num_sentence: int=1000000000,
        generate: Optional[bool] = False,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.tokenizer=tokenizer
        #self.block_size = args.block_size - 2 if generate==False else args.block_size -1
        self.source_block_size = args.source_block_size
        self.target_block_size = args.target_block_size
        self.bos_token_id=self.tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
        self.eos_token_id=self.tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        self.pad_token_id=self.tokenizer.convert_tokens_to_ids(tokenizer.pad_token)


        self.keyword_token_id=self.tokenizer.convert_tokens_to_ids("<keyword>")
        self.length_token_id=self.tokenizer.convert_tokens_to_ids("<length>")

        self.examples = []
        self.data={}

        if "json" in file_path:
            with open(file_path, encoding="utf-8") as f:
                original_data=json.load(f)

        original_data=original_data[0:args.data_size]

        element_names=["source", "target"]
        for element_name in element_names:
            self.data[element_name]=[sentence.get(element_name,"") for sentence in original_data]

            self.data[element_name+"_tokenized"]= \
                [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence)) \
                        for sentence in self.data[element_name]]

        element_keyword_names=["keyword"]
        for element_name in element_keyword_names:
            self.data[element_name]=[sentence.get(element_name,[]) for sentence in original_data]

            self.data[element_name+"_tokenized"]= \
                [[self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(keyword)) \
                    for keyword in keyword_list] \
                        for keyword_list in self.data[element_name]]


        element_names=["length", "position", "relative_position"]
        for element_name in element_names:
            self.data[element_name]=[sentence.get(element_name,"") for sentence in original_data]


        self.examples=[0]*len(original_data)

    def input_generate(self, args, generate=False):
        
        self.examples=[]

        keyword_pos_use_prob=args.keyword_pos_use_prob
        length_use_prob=args.length_use_prob

        keyword_list=[]
        if args.use_keyword:
            keyword_list+=[   
                {
                    "name":["keyword"],
                    "token_id": self.keyword_token_id,
                    "max_num_token": 3,
                }
            ]

        for i in range(len(self.data["source"])):
            source=self.data["source"][i]
            target=self.data["target"][i]
            target_length=self.data["length"][i]
            

            source_tokenized=self.data["source_tokenized"][i]
            target_tokenized=self.data["target_tokenized"][i]

            use_length_judge= \
                (random.random()<length_use_prob or generate)


            keyword_tokens=[]

            for keyword_dict in keyword_list:

                text_keyword = self.data[keyword_dict["name"][0]][i]
                text_keyword_tokenized = self.data[keyword_dict["name"][0]+"_tokenized"][i]
                text_keyword_position=self.data["position"][i]
                text_keyword_relative_position=self.data["relative_position"][i]

                token_id=keyword_dict["token_id"]
                max_num_token=keyword_dict["max_num_token"]

                #When training, choice 0-3 keywords from candidates.
                #When inference, choice 1 keyword from candidates.
                #If you use input_all_keyword option, input all keyword candidates. 
                if args.input_all_keyword==False:

                    num_keyword = \
                        min(int(np.random.rand()*(max_num_token+1)), len(text_keyword_tokenized)) \
                            if generate==False else \
                        min(1, len(text_keyword_tokenized))

                    random_positions=random.sample(list(range(len(text_keyword_tokenized))), num_keyword)
                    text_keyword=[text_keyword[p] for p in random_positions]
                    text_keyword_tokenized=[text_keyword_tokenized[p] for p in random_positions]
                    text_keyword_position=[text_keyword_position[p] for p in random_positions]
                    text_keyword_relative_position=[text_keyword_relative_position[p] for p in random_positions]

                text_keyword_removed=[]
                text_keyword_tokenized_removed=[]
                text_keyword_position_removed=[]
                text_keyword_relative_position_removed=[]
                for k,t,p,rp in zip(text_keyword, text_keyword_tokenized, text_keyword_position, text_keyword_relative_position):
                    find_flag=False
                    for k2, t2 in zip(text_keyword_removed, text_keyword_tokenized_removed):
                        if k in k2 or k2 in k:
                            find_flag=True
                            break
                    if find_flag==False:
                        text_keyword_removed.append(k)
                        text_keyword_tokenized_removed.append(t)
                        text_keyword_position_removed.append(p)
                        text_keyword_relative_position_removed.append(rp)
                text_keyword=text_keyword_removed
                text_keyword_tokenized=text_keyword_tokenized_removed
                text_keyword_position=text_keyword_position_removed
                text_keyword_relative_position=text_keyword_relative_position_removed       

                for keyword, keyword_tokenized, keyword_position, keyword_relative_position \
                in zip(text_keyword, text_keyword_tokenized, text_keyword_position, text_keyword_relative_position):

                    use_position_judge = \
                        random.random()<keyword_pos_use_prob
                        
                    #when specify relative position
                    if args.use_keyword_pos and ((generate and args.generate_keyword_position=="target")
                        or use_position_judge):                        
                        processed_keyword_position=\
                            int(keyword_relative_position*(100/args.position_sep_num))*args.position_sep_num

                    #when specify absolute position
                    elif args.use_keyword_abspos and ((generate and args.generate_keyword_position=="target")
                        or use_position_judge):
                        processed_keyword_position=\
                            int(keyword_position/args.position_abs_sep_num)*args.position_abs_sep_num

                    else:
                        processed_keyword_position="None"


                    keyword_tokens.append((\
                        token_id,
                        keyword_tokenized,
                        self.tokenizer.convert_tokens_to_ids(f"<keyword_pos_{processed_keyword_position}>")))


            keyword_tokens= \
                list(itertools.chain.from_iterable(\
                    [[sep_token_id] \
                    +keyword_token \
                    +([keyword_position])
                        for sep_token_id, keyword_token, keyword_position in keyword_tokens]))


            if args.use_length:
                if use_length_judge==False:
                    text_length="None"
                elif generate and args.generate_text_length=="none":
                    text_length="None"
                elif generate and args.generate_text_length=="target":
                    text_length=int(target_length/args.length_sep_num)*args.length_sep_num
                else:
                    text_length=int(target_length/args.length_sep_num)*args.length_sep_num

                length_tokens= \
                    [self.tokenizer.convert_tokens_to_ids(f"<length_{text_length}>")]                  

            else:
                length_tokens=[]

            #Bart(Enc-Dec model)
            if args.model_type=="encdec":

                source_tokenized=source_tokenized[:self.source_block_size-2-len(keyword_tokens)-len(length_tokens)]
                source_pad_length=\
                    self.source_block_size-2-len(source_tokenized)-len(keyword_tokens)-len(length_tokens)
                target_tokenized=target_tokenized[:self.target_block_size-2]
                target_pad_length=self.target_block_size-2-len(target_tokenized)

                input_ids= \
                    length_tokens \
                    +keyword_tokens \
                    +[self.bos_token_id]+source_tokenized+[self.eos_token_id] \
                    +[self.pad_token_id]*source_pad_length

                head_ids=[]

                decoder_input_ids= \
                    [self.bos_token_id]+target_tokenized+[self.eos_token_id]+ \
                    [self.pad_token_id]*target_pad_length

                target_ids= \
                    [self.bos_token_id]+target_tokenized+[self.eos_token_id]+ \
                    [args.ignore_index]*target_pad_length


                attention_mask= \
                    [1]*(self.source_block_size-source_pad_length) + [0]*source_pad_length


                decoder_attention_mask= \
                    [1]*(self.target_block_size-target_pad_length) + [0]*target_pad_length


            #GPT(decoder-only model)
            elif args.model_type=="dec":
                target_tokenized=\
                    target_tokenized[:self.target_block_size-2-len(keyword_tokens)-len(length_tokens)]
                target_pad_length=\
                    self.target_block_size-2-len(target_tokenized)-len(keyword_tokens)-len(length_tokens)

                if generate==False:
                    input_ids= \
                        length_tokens \
                        +keyword_tokens \
                        +[self.bos_token_id]+target_tokenized+[self.eos_token_id] \
                        +[self.pad_token_id]*target_pad_length
                    attention_mask= \
                        [1]*(self.target_block_size-target_pad_length) + [0]*target_pad_length

                else:
                    input_ids= \
                        length_tokens \
                        +keyword_tokens \
                        +[self.bos_token_id] 
                    attention_mask= \
                        [1]*(self.target_block_size-target_pad_length) + [0]*target_pad_length


                head_ids=[]

                decoder_input_ids = []
                decoder_attention_mask= []

                target_ids= \
                    [args.ignore_index]*len(length_tokens) \
                    +[args.ignore_index]*len(keyword_tokens) \
                    +[args.ignore_index]+target_tokenized+[self.eos_token_id] \
                    +[args.ignore_index]*target_pad_length


            self.examples.append(
                {"input_ids":input_ids,
                "head_ids":head_ids,
                "decoder_input_ids":decoder_input_ids,
                "target_ids":target_ids,
                "attention_mask":attention_mask,
                "decoder_attention_mask":decoder_attention_mask}
                )


        for i in range(0):
            print("input:", self.examples[i]["input_ids"])
            print("target:", self.examples[i]["target_ids"])
            print("decoder_input_ids:", self.examples[i]["decoder_input_ids"]) 
            print("attention:", self.examples[i]["attention_mask"])   



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:  
        return {k:torch.tensor(v, dtype=torch.long) for k,v in self.examples[i].items()}


def main():
    parser = argparse.ArgumentParser()    

    parser.add_argument("--train_data_file", type=str, default=None, help="train data path")
    parser.add_argument("--eval_data_file", type=str, default=None, help="eval data path")
    parser.add_argument("--test_data_file", type=str, default=None, help="test data path")
    parser.add_argument("--min_sentence_size", type=int, default=-1, help="min sentence size")
    parser.add_argument("--source_block_size", type=int, default=1024, help="max sentence size")
    parser.add_argument("--target_block_size", type=int, default=128, help="max sentence size")
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank of logger")

    parser.add_argument("--output_dir", type=str, default="./model/fine_tuned_models/tmp", help="output dir")
    parser.add_argument("--do_train", action="store_true", help="do training")
    parser.add_argument("--do_eval", action="store_true", help="do evaluation")
    parser.add_argument("--do_generate", action="store_true", help="do generation of the sample texts")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="train batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="eval and batch size")
    parser.add_argument("--per_device_generate_batch_size", type=int, default=8, help="generation batch size")
    parser.add_argument("--total_batch_size", type=int, default=256, help="train batch size")
    parser.add_argument("--per_device_pretrain_batch_size", type=int, default=8, help="eval and batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="graddient accumulation steps. automatically decided")
    parser.add_argument("--model_scratch", action="store_true", help="reset pre-trained model params")
    parser.add_argument("--cache_dir", type=str, default=None, help="cache dir of the model")
    parser.add_argument("--init_word_embedding", action="store_true", help="reset word embedding params")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--new_learning_rate", type=float, default=1e-4, help="learning rate for initial emb")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="optimizer params")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="optimizer params")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="optimizer params")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="clip large gradient")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="epochs")
    parser.add_argument("--eval_freq", type=int, default=1, help="eval frequent")

    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--n_gpu", type=int, default=1, help="gpu num")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--fp16", action="store_true", help="use fp16")
    parser.add_argument("--ignore_index", type=int, default=-100, 
    help="ignore index of the crossentropyloss")
    parser.add_argument("--data_size", type=int, default=100000000000000, help="data size")

    
    parser.add_argument("--model_name_or_path", type=str, default=None, help="model name or path")
    parser.add_argument("--model_type", type=str, default="encdec", help="model type. encdec/dec")
    parser.add_argument("--config_name", type=str, default=None, help="config name")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="tokenizer name. if not specified, use tokenizer depending on model")
    parser.add_argument("--use_keyword", action="store_true", help="use keyword to specify sentence. please turn on")
    parser.add_argument("--input_all_keyword", action="store_true", help="Use only keyword input file when in generations")
    parser.add_argument("--use_length", action="store_true", help="Use length to specify sentence")
    parser.add_argument("--use_keyword_weight", action="store_true", help="Use heavey weight of keyword loss")
    parser.add_argument("--use_keyword_pos", action="store_true", help="use keyword position to specify sentence")
    parser.add_argument("--use_keyword_abspos", action="store_true", help="use keyword absolute position to specify sentence")
    parser.add_argument("--length_sep_num", type=int, default=10)
    parser.add_argument("--position_sep_num", type=int, default=10)
    parser.add_argument("--position_abs_sep_num", type=int, default=3)
    parser.add_argument("--keyword_pos_use_prob", type=float, default=0.95)
    parser.add_argument("--length_use_prob", type=float, default=0.95)
    parser.add_argument("--label_smoothing", type=float, default=0.1, 
    help="label smoothing param for loss function")

    
    parser.add_argument("--generation_method", type=str, default="beam")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="num sentence to generate per one prompt")
    parser.add_argument("--filter_keyword_sentence", action="store_true", help="filter sentence that dont include keyword. in depelopment")
    parser.add_argument("--temperature", type=float, default=0.4, help="temperature of genration")
    parser.add_argument("--generate_text_length", type=str, default="target", help="specify sentence length when generation")
    parser.add_argument("--generate_keyword_position", type=str, default="target", help="specify keyword position when generation")
    parser.add_argument("--save_generation", action="store_true", help="Save generated texts")
    parser.add_argument("--num_print_generated_texts", type=int, default=-1, help="how many print generated texts")
    
    parser.add_argument("--num_beams", type=int, default=4, help="beam size")
    parser.add_argument("--length_penalty", type=float, default=2.0, help="length penalty")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="no_repeat_ngram_size")
    parser.add_argument("--min_length", type=int, default=55, help="min length")
    parser.add_argument("--max_length", type=int, default=140, help="max length")
    parser.add_argument("--dataset_type", type=str, default=None, help="cnndm/xsum/stories")



    # See all possible arguments in src/transformers/args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    #parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    #args, args, args = parser.parse_args_into_dataclasses()
    args = parser.parse_args()


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    logger.info("Training/evaluation parameters %s", args)

    # Set seed
    set_seed(args.seed)

    #for key, value in generate_option[args.dataset_type]:
        
    
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    #config
    if args.config_name is None:
        args.config_name=args.model_name_or_path
    config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    # In the version of the bart-large we used for our experiments, this parameter is not set (None).
    # Howerver, in the latest version, it is set to 0. 
    # For reproducibility, we set this parameter to None. It probably has little effect on performance.
    config.forced_bos_token_id = None

    #model & tokenizer
    if args.tokenizer_name is None:
        args.tokenizer_name=args.model_name_or_path

    if args.model_type=="encdec":
        tokenizer = BartTokenizer.from_pretrained(args.tokenizer_name)
        model = BartForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    elif args.model_type=="dec":
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        model = GPT2LMHeadModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

        tokenizer.add_special_tokens({"pad_token":"<|padoftext|>"})
        tokenizer.add_special_tokens({"bos_token":"<|startoftext|>"})
        config.pad_token_id=tokenizer.pad_token_id
        config.bos_token_id=tokenizer.bos_token_id


    tokenizer.add_special_tokens({"additional_special_tokens":[
        "<keyword>","<length>"
    ]})

    for length_i in range(0,700):
        tokenizer.add_special_tokens({"additional_special_tokens":[f"<length_{length_i}>"]})
    tokenizer.add_special_tokens({"additional_special_tokens":["<length_None>"]})

    if args.use_keyword_pos or args.use_keyword_abspos:
        for keyword_pos in range(0,300):
            tokenizer.add_special_tokens({"additional_special_tokens":[f"<keyword_pos_{keyword_pos}>"]})
    tokenizer.add_special_tokens({"additional_special_tokens":["<keyword_pos_None>"]})


    model.resize_token_embeddings(len(tokenizer))

    model=model.to(args.device) 

    if torch.cuda.device_count()>1:
        model=torch.nn.DataParallel(model)
    args.per_device_train_batch_size*=torch.cuda.device_count()
    args.per_device_eval_batch_size*=torch.cuda.device_count()
    args.per_device_pretrain_batch_size*=torch.cuda.device_count()
    args.gradient_accumulation_steps=int(args.total_batch_size/args.per_device_train_batch_size)


    logger.info(f"train batch size: {args.per_device_train_batch_size}, \
        gradient_accumulation_steps: {args.gradient_accumulation_steps}")

    # Get datasets
    train_dataset = (
        TextDataset(args, tokenizer=tokenizer, file_path=args.train_data_file) 
        if args.do_train else None
    )
    eval_dataset = (
        TextDataset(args, tokenizer=tokenizer, file_path=args.eval_data_file)
        if args.do_eval else None
    )
    generate_dataset = (
        TextDataset(args, tokenizer=tokenizer, file_path=args.eval_data_file, generate=True)
        if args.do_generate else None
    )
    
    if args.do_eval:
        eval_dataset.input_generate(args) 
    if args.do_generate:
        generate_dataset.input_generate(args, generate=True) 

    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, collate_fn=None, batch_size=args.per_device_eval_batch_size) \
            if args.do_eval else None
    generate_dataloader = DataLoader(
        generate_dataset, shuffle=False, collate_fn=None, batch_size=args.per_device_generate_batch_size) \
            if args.do_generate else None

    logger.info("train_dataset_size: {}, eval_dataset_size: {}, test_dataset_size: {}"
        .format(len(train_dataset) if args.do_train else None, 
                len(eval_dataset) if args.do_eval else None,
                len(generate_dataset) if args.do_generate else None))

    # Optimizer, Scheduler
    
    total_steps= \
            int(math.ceil(len(train_dataset)/ \
            (args.per_device_train_batch_size*args.gradient_accumulation_steps))* \
            args.num_train_epochs) if args.do_train else 0

    warmup_steps=math.ceil(total_steps*0.06)
    logger.info(f"total steps: {total_steps}, warmup_steps: {warmup_steps}")

    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    #shared
    word_emb_layer = \
        "wte" if "gpt" in args.model_name_or_path else \
        "shared" if "bart" in args.model_name_or_path else \
        "xxxxxxxxxxxxxx"
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, "lr":args.learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if word_emb_layer not in n and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, "lr":args.learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if word_emb_layer in n and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, "lr":args.new_learning_rate,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_fct=torch.nn.CrossEntropyLoss(
        ignore_index=args.ignore_index, label_smoothing=args.label_smoothing)
    
    os.makedirs(args.output_dir, exist_ok=True)

    def train(num_epoch=args.num_train_epochs):
                
        for epoch in range(int(num_epoch)):
            logger.info("start epoch {}".format(epoch))
            train_dataset.input_generate(args)
            train_dataloader = DataLoader(
                train_dataset, shuffle=True, collate_fn=None, batch_size=args.per_device_train_batch_size)
            model.train()
            for step, batch in enumerate(train_dataloader):
                
                input_ids=batch["input_ids"].to(args.device)
                decoder_input_ids=batch["decoder_input_ids"].to(args.device)
                target_ids=batch["target_ids"].to(args.device)
                attention_mask=batch["attention_mask"].to(args.device)
                decoder_attention_mask = batch["decoder_attention_mask"].to(args.device)

                if args.model_type=="encdec":
                    outputs=model(
                        input_ids=input_ids, 
                        decoder_input_ids=decoder_input_ids,
                        attention_mask=attention_mask,
                        decoder_attention_mask=decoder_attention_mask)[0]
                elif args.model_type=="dec":
                    outputs=model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask)[0]              

                #loss
                outputs = outputs[:, :-1, :].contiguous()
                target_ids = target_ids[:, 1:].contiguous()
                loss=loss_fct(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))

                if torch.cuda.device_count()>1:
                    loss = loss.mean()
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if (step+1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
            if (epoch+1)%args.eval_freq==0:
                evaluate()
    
    def evaluate():
        model.eval()
        losses = []
        for step, batch in tqdm(enumerate(eval_dataloader)):
            with torch.no_grad():
                input_ids=batch["input_ids"].to(args.device)
                decoder_input_ids=batch["decoder_input_ids"].to(args.device)
                target_ids=batch["target_ids"].to(args.device)
                attention_mask=batch["attention_mask"].to(args.device)
                decoder_attention_mask = batch["decoder_attention_mask"].to(args.device)

                if args.model_type=="encdec":
                    outputs=model(
                        input_ids=input_ids, 
                        decoder_input_ids=decoder_input_ids,
                        attention_mask=attention_mask,
                        decoder_attention_mask=decoder_attention_mask)[0]
                elif args.model_type=="dec":
                    outputs=model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask)[0]

                #loss
                outputs = outputs[..., :-1, :].contiguous()
                target_ids = target_ids[..., 1:].contiguous()
    
            loss = loss_fct(outputs.view(-1, outputs.size(-1)), target_ids.view(-1)).view(-1)
            losses.append(loss)

        losses = torch.cat(losses)
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(args.output_dir, "eval_results_lm.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    def generate():

        generate_option = {
            "cnndm": {"num_beams":4, "min_length":55, "max_length":140, 
                "no_repeat_ngram_size":3, "length_penalty":2.0},
            "xsum": {"num_beams":6, "min_length":10, "max_length":60, 
                "no_repeat_ngram_size":3, "length_penalty":1.0},
            "stories": {},
        }[args.dataset_type]

        if "num_beams" in generate_option and args.num_return_sequences>generate_option["num_beams"]:
            generate_option["num_beams"]=args.num_return_sequences

        model.eval()

        generated_data=[]
        bos_token_id = generate_dataset.bos_token_id
        eos_token_id = generate_dataset.eos_token_id

        for step, batch in tqdm(enumerate(generate_dataloader)):
            with torch.no_grad():
                input_ids = batch["input_ids"].to(args.device)
                attention_mask=batch["attention_mask"].to(args.device)
                if args.model_type=="encdec":
                    outputs=model.generate(
                        input_ids=input_ids, 
                        attention_mask=attention_mask,
                        num_return_sequences=args.num_return_sequences,
                        **generate_option
                    )

                    control_ids = []
                    for text in input_ids:
                        bos_index=[i for i in range(len(text)) if text[i]==bos_token_id][0]
                        control_ids.append(text[:bos_index])
                    outputs = outputs      

                elif args.model_type=="dec":
                    outputs=model.generate(
                        input_ids=input_ids,
                        do_sample=True,
                        top_p=0.95, 
                        max_length=128, 
                        temperature=args.temperature,
                        no_repeat_ngram_size=None,
                        num_return_sequences=args.num_return_sequences)

                    control_ids = []
                    for text in outputs:
                        bos_index=[i for i in range(len(text)) if text[i]==bos_token_id][0]
                        control_ids.append(text[:bos_index])
                    outputs = [text[bos_index:] for text in outputs]
            
            start = step * args.per_device_generate_batch_size
            end = (step+1) *args.per_device_generate_batch_size 
            hypothesis=tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            reference=\
                list(itertools.chain.from_iterable(\
                    [[text]*args.num_return_sequences for text in generate_dataset.data["target"][start:end]]
                ))
            
            for idx in range(len(hypothesis)):
                control_tokens = [control_ids[int(idx/args.num_return_sequences)]]

                control_tokens = tokenizer.batch_decode(
                    control_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
                    
                control_tokens_split = control_tokens.split("<keyword>")

                if args.use_length:
                    length=re.search("<length_(.+?)>", control_tokens_split[0]).group(1)
                else:
                    length="None"

                keywords=[]
                for tokens in control_tokens_split[1:]:
                    match = re.search("(.*)<keyword_pos_(.+?)>", tokens)
                    keywords.append({
                        "token":match.group(1).strip().split(), 
                        "position":match.group(2),
                    })
                    
                generated_data.append({
                    "reference": reference[idx],
                    "hypothesis": hypothesis[idx],
                    "reference_length": length,
                    "reference_keyword": keywords,
                })


        with open(os.path.join(args.output_dir, "generated_data.json"), "w", encoding="utf-8") as f:
            json.dump(generated_data, f, ensure_ascii=False, indent=4)  


    if args.do_train:
        progress_bar = tqdm(range(total_steps))
        logger.info("*** Init Evaluate ***")
        evaluate()

        # Training
        if args.num_train_epochs>0:
            logger.info("*** Train ***")
            train(num_epoch=args.num_train_epochs)


    # Evaluation
    if args.do_train==False and args.do_eval and args.do_generate==False:
        logger.info("Evaluate")
        evaluate()

    if torch.cuda.device_count()>1:
        model=model.module
        args.device="cuda:0"
        model=model.to(args.device)

    # Generation
    if args.do_generate:
        logger.info("Generate")        
        generate()

    #model save
    if args.do_train and args.output_dir is not None:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir) 
        
    #return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

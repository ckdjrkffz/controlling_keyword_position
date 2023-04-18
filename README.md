## Overview
This is code for an experiment to reproduce the paper. During the review, refer to this code if necessary.

## Requirement
```
pip install -r requirements.txt
```

## Data Preprocess

#### CNN/DM

Get the dataset from [here](https://github.com/icml-2020-nlp/semsim/tree/master/datasets) and extract the dataset. Next, run following preprocess script.

```
python prepare_cnndm.py --data_path dataset/cnndm
```

#### XSum

Since the dataset was not available from the original [github repository](https://github.com/EdinburghNLP/XSum), we used the huggingface functionality to acquire it instead.

```
python prepare_xsum.py --data_path dataset/xsum
```


#### ROCStories

Get this dataset from [site](https://cs.rochester.edu/nlp/rocstories/).

```
python prepare_stories.py --data_path dataset/stories
```

(For the following description, we will only discuss the CNN/DM dataset basically. The same process should be followed for other data.)

#### Control tokens extraction

To extract control tokens (keyword, keyword positions, text length) to control text generation, run following scripts.

- Train: 1-3 keyword, Test: 1 keyword
```
python prepare_keyword_extraction.py --data_path dataset/cnndm
```

- Train: 1-3 keyword, Test: n keyword
```
python prepare_keyword_extraction.py --data_path dataset/cnndm --num_keywords $n
```



## Model 

We got the BART-large model from [here](https://huggingface.co/facebook/bart-large).
We got the GPT model from [here](https://huggingface.co/gpt2).

Instead of downloading the model directly, you may specify the model name directly in `--model_name_or_path` option in the following code. For more information, please refer to the Huggingface documentation or other sources.


## Train & Inference

### CNN/DM

#### Train
```
python run_conditional_modeling.py \
    --model_name_or_path ./model/pre_trained_models/bart-large \
    --do_train --do_eval \
    --train_data_file dataset/cnndm/train.json \
    --eval_data_file dataset/cnndm/val.json \
    --output_dir ./model/fine_tuned_models/cnndm_bart_lsep5_psep10_e10_lr2e-5_b32_seed0 \
    --num_train_epochs 10 --learning_rate 2e-5 --new_learning_rate 1e-3 --seed 0 \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 16 --total_batch_size 32 \
    --use_length --use_keyword --use_keyword_pos \
    --length_sep_num 5 --position_sep_num 10 --label_smoothing 0.1 \
    --keyword_pos_use_prob 0.90 --length_use_prob 0.90
```

#### Generate

```
python run_conditional_modeling.py \
    --model_name_or_path ./model/fine_tuned_models/cnndm_bart_lsep5_psep10_e10_lr2e-5_b32_seed0 \
    --do_generate \
    --eval_data_file dataset/cnndm/test.json \
    --output_dir out/cnndm_bart_lsep5_psep10_e10_lr2e-5_b32_seed0 \
    --dataset_type cnndm \
    --per_device_generate_batch_size 32 \
    --use_length --use_keyword --use_keyword_pos \
    --length_sep_num 5 --position_sep_num 10 \
    --seed 0 --input_all_keyword
```

### XSum

#### Train
```
python run_conditional_modeling.py \
    --model_name_or_path ./model/pre_trained_models/bart-large \
    --do_train --do_eval \
    --train_data_file dataset/xsum/train.json \
    --eval_data_file dataset/xsum/val.json \
    --output_dir ./model/fine_tuned_models/xsum_bart_lsep5_psep10_e10_lr2e-5_b32_seed0 \
    --num_train_epochs 10 --learning_rate 2e-5 --new_learning_rate 1e-3 --seed 0 \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 16 --total_batch_size 32 \
    --use_length --use_keyword --use_keyword_pos \
    --length_sep_num 5 --position_sep_num 10 --label_smoothing 0.1 \
    --keyword_pos_use_prob 0.90 --length_use_prob 0.90
```

#### Generate

```
python run_conditional_modeling.py \
    --model_name_or_path ./model/fine_tuned_models/xsum_bart_lsep5_psep10_e10_lr2e-5_b32_seed0 \
    --do_generate \
    --eval_data_file dataset/xsum/test.json \
    --output_dir out/xsum_bart_lsep5_psep10_e10_lr2e-5_b32_seed0 \
    --dataset_type xsum \
    --per_device_generate_batch_size 32 \
    --use_length --use_keyword --use_keyword_pos \
    --length_sep_num 5 --position_sep_num 10 \
    --seed 0 --input_all_keyword
```


### ROCStories

#### Train
```
python run_conditional_modeling.py \
    --model_name_or_path ./model/pre_trained_models/gpt2 \
    --model_type dec \
    --do_train --do_eval \
    --train_data_file dataset/stories/train.json \
    --eval_data_file dataset/stories/val.json \
    --output_dir ./model/fine_tuned_models/stories_gpt_lsep5_psep10_e30_lr2e-5_b32_seed0 \
    --num_train_epochs 30 --learning_rate 2e-5 --new_learning_rate 1e-3 --seed 0 \
    --target_block_size 80 \
    --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --total_batch_size 32 \
    --use_length --use_keyword --use_keyword_pos \
    --length_sep_num 5 --position_sep_num 10 --label_smoothing 0.1 \
    --keyword_pos_use_prob 0.90 --length_use_prob 0.90
```

#### Generate

```
python run_conditional_modeling.py \
    --model_name_or_path ./model/fine_tuned_models/stories_gpt_lsep5_psep10_e30_lr2e-5_b32_seed0 \
    --model_type dec \
    --do_generate \
    --eval_data_file dataset/stories/test.json \
    --output_dir out/stories_gpt_lsep5_psep10_e30_lr2e-5_b32_seed0 \
    --dataset_type stories \
    --per_device_generate_batch_size 1 \
    --use_length --use_keyword --use_keyword_pos \
    --length_sep_num 5 --position_sep_num 10 --temperature 0.1 --generation_method sample \
    --seed 0 --input_all_keyword
```


## Eval the result

#### Keyword Includion & Position control (Simple analysis, it can be used for multiple keyword setting)

```
python eval_keyword_correct.py \
--data_path out/cnndm_bart_lsep5_psep10_e10_lr2e-5_b32_seed0
```

#### Keyword Includion & Position control (Detail analysis, only for one keyword)

```
python eval_generated_texts.py \
--data_path out/cnndm_bart_lsep5_psep10_e10_lr2e-5_b32_seed0
```


#### Summarization eval

We use [rouge-score](https://pypi.org/project/rouge-score/) to eval ROUGE score.

```
python eval_summarization.py \
--data_path out/cnndm_bart_lsep5_psep10_e10_lr2e-5_b32_seed0
```


## License
Apache 2.0
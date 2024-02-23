import pandas as pd
import numpy as np
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, AdamW
import tqdm

# cuda 사용여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f'Using device : {device}')


# data preprocessing
# 데이터 로드
data = pd.read_csv('train.csv')

# 토크나이저 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/logpt2-base-v2', eos_token='</s>')

# 데이터 포맷팅 및 토크나이징
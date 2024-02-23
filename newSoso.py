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
formatted_data = []
for _ , row in tqdm(data.iterrows()):
    for q_col in ['질문_1', '질문_2']:
        for a_col in ['답변_1', '답변_2', '답변_3', '답변_4', '답변_5']:
            #질문과 답변 쌍을 </s> token으로 연결
            input_text = row[q_col] + tokenizer.eos_token + row[a_col]
            input_ids = tokenizer.encode(input_text, return_tensors= 'pt')
            formatted_data.append(input_ids)
print('Done.')

# 모델로드
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
model.to(device)
# 모델 학습 하이퍼파라미터(Hyperparameter) 세팅
# 실제 필요에 따라 조정
CFG = {
    'LR' : 2e-5, # Learning Rate
    'EPOCHS' : 10 # 학습 Epoch
}

# 모델 학습 설정
optimizer = AdamW(model.parameters(), lr=CFG['LR'])
model.train()

# 모델 학습
for epoch in range(CFG['EPOCHS']):
    total_loss = 0
    progress_bar = tqdm(enumerate(formatted_data), total=len(formatted_data))
    for batch_idx, batch in progress_bar:
        
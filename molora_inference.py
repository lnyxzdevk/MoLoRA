# REQUIRE LATEST PEFT LIBRARY
import torch
import datetime
import random
import os
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from huggingface_hub import hf_hub_download
from utils.prompter import Prompter
from tqdm import tqdm

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(42)

print('>> Embedding Model Loading...')
embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
print('>> Load Success.')

print('>> Root Model Loading...')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-13b-hf', load_in_8bit=True, torch_dtype=torch.float16, device_map='auto')
model = PeftModel.from_pretrained(model, 'crowdworks/cw-llama2-13b', adapter_name='default')
print('>> Load Success.')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf')
centers = torch.load(hf_hub_download(repo_id='crowdworks/EvolInstruct-cluster', filename='centers.pt', repo_type='dataset')).cuda()

prompter = Prompter('alpaca')
text = '대한민국의 대표적인 유물에 대해 알려줘.'

prompt = prompter.generate_prompt(text)
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to('cuda')
inputs = {k:v.cuda() for k, v in tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).items()}
print('>> Input Prompt:')
print(prompt)

print('>> Original Root Model Response Generating...')
model.set_adapter('default')
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_k=20,
    top_p=.9,
    do_sample=True,
)
print(tokenizer.decode(outputs[0]))

print('>> Cluster Adapter Loading...')
for i in tqdm(range(1, 5)):
    model.load_adapter(f'./models/llama-13b-hf-c{str(i)}', adapter_name=f'cluster-{str(i)}')
print('>> Load Success.')

print('>> Cluster Adapter Selection Decision... ')
prompt_embeddings = torch.tensor(embedding_model.encode([text]), device='cuda')
merge_temperature = 2
merge_alpha = 16
cluster_alphas= torch.softmax(
    torch.nn.functional.cosine_similarity(prompt_embeddings, centers) * merge_temperature, -1
).mul(merge_alpha // 4).tolist()
print(f'>> Decision Success.')
print(f'>> Alpha for each cliusters: {cluster_alphas}')

print(f'>> Adapter Combining...')
unique_name = str(hash(datetime.datetime.now()))
model.add_weighted_adapter([f'cluster-{str(i)}' for i in range(1, 5)], cluster_alphas, combination_type='svd', adapter_name=unique_name)
model.set_adapter(unique_name)
print(f'>> Combine Success')

model.save_pretrained(save_directory='./cluster_adapter', selected_adapters=[unique_name])

print('>> Response Example Generating...')
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_k=20,
    top_p=.9,
    do_sample=True,
)

print(tokenizer.decode(outputs[0]))

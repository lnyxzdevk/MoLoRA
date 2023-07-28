import os
import sys
from typing import List
from copy import deepcopy
import fire
import torch
import transformers
import logging
import random
import numpy as np

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, TrainerCallback, GenerationConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from utils.prompter import Prompter

#logger = logging.getLogger(__name__)

class ColorFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__(fmt)
        self.FORMATS = {
            logging.DEBUG: self.grey + fmt + self.reset,
            logging.INFO: self.blue + fmt + self.reset,
            logging.WARNING: self.yellow + fmt + self.reset,
            logging.ERROR: self.red + fmt + self.reset,
            logging.CRITICAL: self.bold_red + fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def set_file_logger(name, dir, use_console=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    os.makedirs(dir, exist_ok=True)

    if use_console:
        logger.propagate = False # disable default handler
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setLevel(logging.INFO)
        consoleHandler.setFormatter(ColorFormatter("%(asctime)s | %(levelname)s %(message)s"))
        logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(os.path.join(dir,'session.log'), mode='a') 
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s %(message)s"))
    logger.addHandler(fileHandler)
    return logger

def train(
        # 모델, 데이터 파라미터
        base_model: str,
        data_path: str,
        output_dir: str,
        cluster_number: int,
        model_type: str = 'llama',
        # 학습 하이퍼파라미터
        batch_size: int = 128,
        micro_batch_size: int = 2, # RTX3090
        num_epochs: int = 3,
        learning_rate: int = 3e-4,
        cutoff_len: int = 1024,
        val_set_size: int = 0.05,
        # LoRA 하이퍼파라미터
        lora_r: int = 8, # 보통 lora_alpha / 2
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # LLM 하이퍼파라미터
        train_on_inputs: bool = True,
        add_eos_token: bool = False,
        group_by_length: bool = False,
        # Wandb 파라미터
        wandb_project: str = None,
        wandb_run_name: str = None,
        wandb_watch: str = None,
        wandb_log_model: str = None,
        resume_from_checkpoint: str = None,
        prompt_template_name: str = 'ko_alpaca',
):
    logger = set_file_logger(__name__, output_dir)
    assert base_model, 'Base model을 설정해주세요.'

    gradient_accumulation_steps = batch_size // micro_batch_size
    prompter = Prompter(prompt_template_name)

    device_map = 'auto'
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    ddp = world_size != 1
    if ddp:
        device_map = {'': int(os.environ.get('LOCAL_RANK') or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    # Wandb 설정
    wandb_project = 'EvolInstruct-CLUSTER'
    wandb_run_name = f'Llama2-13b-c{cluster_number}'


    use_wandb = len(wandb_project) > 0  or ('WANDB_PROJECT' in os.environ and len(os.environ['WANDB_PROJECT']) > 0)
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

    if model_type == 'llama':
        model = LlamaForCausalLM.from_pretrained(base_model, load_in_8bit=True, torch_dtype=torch.float16, device_map=device_map)
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        lora_target_modules=['q_proj', 'v_proj']
    elif model_type == 'polyglot':
        model = AutoModelForCausalLM.from_pretrained(base_model, load_in_8bit=True, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
        lora_target_modules=['query_key_value', 'xxx']
    logger.info(f'>>> Load Model from {base_model}')
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
        if result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len and add_eos_token:
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result
    
    def generate_and_tokenize_prompt(data_point):
        try:
            if len(data_point['input']) != 0:
                full_prompt = prompter.generate_prompt(instruction=data_point["instruction"], input=data_point["input"], label=data_point["output"])
            else:
                full_prompt = prompter.generate_prompt(instruction=data_point["instruction"], label=data_point["output"])
        except:
            full_prompt = prompter.generate_prompt(instruction=data_point['instruction'], label=data_point['output'])
        tokenized_full_prompt = tokenize(full_prompt)
        
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(data_point["instruction"], data_point["input"])
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        
        return tokenized_full_prompt
    
    model = prepare_model_for_int8_training(model)

    
        
    config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules, lora_dropout=lora_dropout, bias='none', task_type='CAUSAL_LM')

    model = get_peft_model(model, config)
    logger.info(f'>>> LoRA Applied.')
    if data_path.endswith('.json') or data_path.endswith('jsonl'):
        data = load_dataset('json', data_files=data_path, split='train')
    else:
        data = load_dataset(data_path, data_files=f'c{cluster_number}.json')

    data = data['train'].select(range(2048))
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, 'pytorch_model.bin')
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, 'adapter_model.bin')
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            print(f'{checkpoint_name}으로부터 다시 시작합니다.')
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f'체크포인트 {checkpoint_name}가 없습니다.')
    
    model.print_trainable_parameters()
    
    #val_set_size = 10
    if val_set_size > 0:
        train_val = data.train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = (train_val['train'].shuffle().map(generate_and_tokenize_prompt))
        val_data = (train_val['test'].shuffle().map(generate_and_tokenize_prompt))
    else:
        train_data = data.shuffle().map(generate_and_tokenize_prompt)
        val_data = None
    
    logger.info(f'>>> Processed data from {data_path}')
    logger.info(f'>>> Using prompt {prompt_template_name}, prompt example \n {tokenizer.decode(train_data[0]["input_ids"])}')

    if not ddp and torch.cuda.device_count() > 1:
        logger.info(f'>>> Not use DDP')
        model.is_parallelizable = True
        model.model_parallel = True
    
    class CustomCallback(TrainerCallback):
        def __init__(self, trainer) -> None:
            super().__init__()
            self.trainer = trainer
            self.generation_config = GenerationConfig(
                temperature=1.0,
                top_p=0.75,
                top_k=40,
                num_beams=2,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=1024, # max_length=max_new_tokens+input_sequence
                min_new_tokens=1, # min_length=min_new_tokens+input_sequence
                bad_words_ids=tokenizer(['\n\n### 명령어:','\n\n### 응답:'], add_special_tokens=False).input_ids
            )
            self.repetition_penalty=1.3
            self.logger = set_file_logger('transformers.trainer', trainer.args.output_dir)

        def on_log(self, args, state, control, logs, **kwargs):
            logger.info(logs)


    args = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=5,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=1,
        optim='adamw_torch',
        evaluation_strategy='steps' if val_set_size > 0 else 'no',
        save_strategy='steps',
        eval_steps=20 if val_set_size > 0 else None,
        save_steps=20,
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True if val_set_size > 0  else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to='wandb' if use_wandb else None,   
        run_name=wandb_run_name if use_wandb else None,
        #deepspeed='deepspeed.json',
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=args,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors='pt', padding=True),
    )
    trainer.add_callback(CustomCallback(trainer))
    model.config.use_cache = False

    if torch.__version__ >= '2' and sys.platform != 'win32':
        model = torch.compile(model)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)

if __name__ == '__main__':
    seed_everything(42)
    fire.Fire(train)

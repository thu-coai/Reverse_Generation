import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import json
from typing import Optional
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, HfArgumentParser
from tqdm import tqdm
import wandb
from random import choice
import os
from utils import get_logger, try_create_dir, batch_to_device, top_k_top_p_filtering

logger = get_logger(__name__)


class MyDataset(Dataset):
    def __init__(self, path, tokenizer, args):
        self.data = pd.read_csv(path)
        # with open(path, encoding='utf-8') as f:
        #     self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.args = args
        self.mode = args.mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode != 'gen':
            if self.args.reverse == True:
                context = self.data.loc[index]['response']
                response = self.data.loc[index]['context']
            else:
                context = self.data.loc[index]['context']
                response = self.data.loc[index]['response']
            kind = self.data.loc[index]['ctx_kind']
            return {'context': context, 'response': response, 'kind': kind}

class Helper:
    all_kinds = ['IDENTITY_ATTACK', 'INSULT', 'PROFANITY',
         'THREAT', 'SEXUALLY_EXPLICIT', 'FLIRTATION', 'drugs', 'religion', 'nsfw', 'politics', 'medical', 'none']

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.kind2idx = {kind:i for i, kind in enumerate(self.all_kinds)}

    
    def collate_fn(self, batch):
        contexts = [f'[{sample["kind"]}]' + sample['context'] +
                    self.tokenizer.eos_token for sample in batch]
        responses = [sample['response'] + self.tokenizer.eos_token for sample in batch]

        context_ids = [self.tokenizer.encode(context, add_special_tokens=False) for context in contexts]
        response_ids = [self.tokenizer.encode(response, add_special_tokens=False) for response in responses]
        concat_ids = [torch.tensor(cid + rid, dtype=torch.long) for cid, rid in zip(context_ids, response_ids)]
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        pad_ids = pad_sequence(concat_ids, batch_first=True, padding_value=pad_token_id)
        labels = torch.full_like(pad_ids, -100)
        for i in range(len(context_ids)):
            start = len(context_ids[i])
            end = start + len(response_ids[i])
            labels[i, start:end] = pad_ids[i, start:end]

        res = {'input_ids': pad_ids, 'labels': labels}
        res['kinds'] = [self.kind2idx[sample['kind']] for sample in batch]

        return res



class ControlCodeModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrain_path)
        self.model = AutoModelForCausalLM.from_pretrained(args.pretrain_path)
        self.num_token = args.num_token
        self.n_layer = self.model.config.n_layer
        self.n_embd = self.model.config.n_embd
        self.mid_dim = args.mid_dim
        self.args = args

        # self.kind_num = len(Helper.all_kinds)
        # if args.use_prefix:
        #     self.kind_prefixes = nn.ModuleList([OneKindPrefix(args, self.model.config) for _ in range(self.kind_num)])

    def fix_lm(self):
        trained_pars = [p for p in self.parameters() if p.requires_grad is True]
        print(f'before fix length:{len(trained_pars)}')
        # for name, p in self.named_parameters():
        #     if p.requires_grad is True:
        #         print(name)

        for p in self.model.parameters():
            p.requires_grad = False

        trained_pars = [p for p in self.parameters() if p.requires_grad is True]
        print(f'after fix length:{len(trained_pars)}')

        # for name, p in self.named_parameters():
        #     if p.requires_grad is True:
        #         print(name)

    def forward(self, input_ids, labels, kinds):
        
        return self.model(input_ids=input_ids, labels=labels)


@dataclass
class Arguments:
    finetunedata_dir = '../dialogpt_data/prefixtune'
    suffix = '_highInduce'

    train_data_path: Optional[str] = field(
        default=f'{finetunedata_dir}/train{suffix}.csv')
    val_data_path: Optional[str] = field(
        default=f'{finetunedata_dir}/validation{suffix}.csv')
    test_data_path: Optional[str] = field(
        default=f'{finetunedata_dir}/validation{suffix}.csv')
    gen_data_path: Optional[str] = field(
        default=f'{finetunedata_dir}/validation_gen.csv')
    gen_out_data_path: Optional[str] = field(
        default=f'{finetunedata_dir}/validation_gen_out.csv')
    lr: Optional[float] = field(default=2e-5)
    gpus: Optional[int] = field(default=1)
    seed: Optional[int] = field(default=2022)
    epochs: Optional[int] = field(default=2)
    train_batch_size: Optional[int] = field(default=8)
    valtest_batch_size: Optional[int] = field(default=32)
    gen_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    mode: Optional[str] = field(default='train')
    ckpt_path: Optional[str] = field(
        default='../logs/dialogpt_prefixtune/1_usePrefixFalse.ckpt')
    pretrain_path: Optional[str] = field(
        default='microsoft/DialoGPT-medium')
    root_dir: Optional[str] = field(default='../logs/dialogpt_controlcode')
    reverse: Optional[bool] = field(default=True)
    load_ckpt: Optional[bool] = field(default=False)
    mid_dim: Optional[int] = field(default=512)
    num_token: Optional[int] = field(default=2)
    prefix_dropout: Optional[float] = field(default=0.)
    use_prefix: Optional[bool] = field(default=True)

    def __post_init__(self):        
        # if os.path.exists(self.ckpt_path):
        #     names = os.listdir(self.ckpt_path)
        #     self.ckpt_path = os.path.join(self.ckpt_path, names[0])
        # train_name = self.train_data_path.split('/')[-1].split('.')[0]
        # self.save_path_suffix = f'_posAlpha{self.pos_alpha}_seed{self.seed}_neg{str(self.negative_train)}_{train_name}'
        # logger.info(f'save path suffix:{self.save_path_suffix}')
        self.save_path_suffix = f'_[kind]ctx_lr{self.lr}'


def data_process(args):
    df = pd.read_csv('../../data_analysis/data/single_turn_high_induce_ctx.csv')
    print(len(df))
    print(df)
    
    if 'Unnamed: 0' in df:
        df.drop(['Unnamed: 0'], axis=1, inplace=True)

    idxs = list(range(len(df)))
    np.random.shuffle(idxs)
    s1 = int(len(df) * 0.8)
    s2 = int(len(df) * 0.9)
    df_train = df.iloc[idxs[:s1]]
    df_val = df.iloc[idxs[s1:s2]]
    df_test = df.iloc[idxs[s2:]]

    out_dir = '../dialogpt_data/prefixtune'

    train_path = f'{out_dir}/train_highInduce.csv'
    val_path = f'{out_dir}/validation_highInduce.csv'
    test_path = f'{out_dir}/test_highInduce.csv'

    df_train.to_csv(train_path)
    df_val.to_csv(val_path)
    df_test.to_csv(test_path)


def interactive(args):
    model = PrefixTuningModel(args)
    tokenizer = model.tokenizer

    device = torch.device("cuda")

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    helper = Helper(tokenizer)

    with torch.no_grad():
        while True:
            UTTERANCE = input("Enter your sentence: ")
            inputs = tokenizer([UTTERANCE + tokenizer.eos_token], return_tensors='pt', truncation=True,
                            padding=True, max_length=tokenizer.model_max_length, add_special_tokens=False)
            batch_to_device(inputs, device)
            if args.use_prefix:
                max_length = 100
                kind = choice(Helper.all_kinds)
                print(f'sampled kind:{kind}')
                pvs = model.kind_prefixes[helper.kind2idx[kind]].get_past_key_values()
                temp = []
                for i in range(2):
                    temp.append(pvs[i].split(1)) # layer * [1, batch=1, head, seqlen, emb]
                l = len(temp[0])
                past_key_values = []
                for i in range(l):
                    x = []
                    for j in range(2):
                        x.append(temp[j][i].squeeze(0))
                    past_key_values.append(x)

                input_ids = inputs['input_ids']
                l = input_ids.size(-1)
                # print(inputs)
                # print(len(past_key_values), len(past_key_values[0]), past_key_values[0][0].size(), past_key_values[0][1].size())
                for i in range(max_length):
                    outputs = model.model(input_ids=input_ids, use_cache=True, past_key_values=past_key_values)
                    logits = outputs.logits
                    past_key_values = outputs.past_key_values
                    logits = top_k_top_p_filtering(logits[:, -1, :], top_k=10)
                    probs = torch.softmax(logits, dim=-1).squeeze(1)
                    token = torch.multinomial(probs, num_samples=1)
                    if token.item() == tokenizer.eos_token_id:
                        break
                    input_ids = torch.cat([input_ids, token], dim=-1)
                    # print(input_ids)

                reply_ids = input_ids[:, l:]
                print(tokenizer.batch_decode(reply_ids, skip_special_tokens=True))

            else:
                reply_ids = model.model.generate(
                    **inputs, num_beams=1, do_sample=True, top_k=30, max_length=100)
                print(tokenizer.batch_decode(reply_ids, skip_special_tokens=True))
        

def main(args):
    pl.seed_everything(args.seed)
    if args.mode == 'data':
        data_process(args)

    elif args.mode == 'interactive':
        interactive(args)

    elif args.mode == 'train':
        model = ControlCodeModel(args)
        device = torch.device("cuda")

        model.train()
        model.to(device)
        
        tokenizer = model.tokenizer
        train_dataset = MyDataset(args.train_data_path, tokenizer, args)
        val_dataset = MyDataset(args.val_data_path, tokenizer, args)
        
        helper = Helper(tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=32,
                                      pin_memory=True, collate_fn=helper.collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=args.valtest_batch_size, shuffle=False, num_workers=32,
                                      pin_memory=True, collate_fn=helper.collate_fn)
                                    
        optimizer = AdamW(filter(lambda p: p.requires_grad,
                          model.parameters()), lr=args.lr)
        
        wandb.init(project='dialogpt_controlcode', name=f'[kind]ctx_lr{args.lr}')
        best_path = None
        best_loss = None
        for epoch in range(args.epochs):
            tqdm_bar = tqdm(enumerate(train_dataloader),
                            desc=f'Epoch {epoch}', total=len(train_dataloader))
            model.train()
            for batch_idx, batch in tqdm_bar:
                batch_to_device(batch, device)
                outputs = model(**batch)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                tqdm_bar.set_postfix({'loss': loss.item()})
                wandb.log({'training loss': loss.item()})
            
            model.eval()
            val_losses = []
            for batch_idx, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Validation: '):
                batch_to_device(batch, device)
                with torch.no_grad():
                    outputs = model(**batch)
                    loss = outputs.loss
                    val_losses.append(loss.item())
            new_loss = np.mean(val_losses)

            print(f'validation loss:{new_loss}')
            wandb.log({'validation loss': new_loss})
            
            save_path = os.path.join(args.root_dir, f'{epoch}{args.save_path_suffix}.ckpt')
            logger.info(f'savepath: {save_path}')

            if best_loss is None:
                best_loss = new_loss
                best_path = save_path
                torch.save(model.state_dict(), save_path)

            elif new_loss < best_loss:
                os.remove(best_path)
                best_loss = new_loss
                best_path = save_path
                torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    try_create_dir('../logs')
    try_create_dir('../dialogpt_data/prefixtune')
    parser = HfArgumentParser(Arguments)
    args, = parser.parse_args_into_dataclasses()
    try_create_dir(args.root_dir)
    print(args)
    main(args)


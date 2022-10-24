import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from prompt_tuning_dialogpt import PromptTuningModel
from utils import top_k_top_p_filtering, repetition_penalty
from pytorch_lightning import seed_everything


def dexpert_gen_bayes(path, outpath, ckpt_path, ae_ckpt_path, bayes_ckpt_path, reverse=True, resp_sample_num=1):
    # in json, out csv

    model_name = 'microsoft/DialoGPT-medium'
    args = Namespace(pretrain_path=model_name, use_prefix=False, num_token=10)
    device = torch.device('cuda')

    def load_tokenizer_model(path):
        plmodel = PromptTuningModel(args)
        plmodel.load_state_dict(torch.load(path))
        model = plmodel.model
        tokenizer = plmodel.tokenizer
        model.to(device)
        model.eval()
        return tokenizer, model

    tokenizer, model = load_tokenizer_model(ckpt_path)
    print(model.config.length_penalty)
    tokenizer, ae_model = load_tokenizer_model(ae_ckpt_path)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    # bayes_tokenizer = AutoTokenizer.from_pretrained(model_name)
    # bayes_model = AutoModelForCausalLM.from_pretrained(model_name)
    # bayes_model.to(device).eval()
    bayes_tokenizer, bayes_model = load_tokenizer_model(bayes_ckpt_path)
    print(f'finish load models!')

    if '.json' in path:
        with open(path) as f:
            a = json.load(f)
    else:
        a = []
        x = pd.read_csv(path)

        for context, response in zip(x['context'], x['response']):
            a.append({'context': context, 'response': response})

    res = []
    for x in a:
        res.extend([x] * resp_sample_num)

    a = res

    contexts = []
    responses = []

    batch_size = 16
    for i in tqdm(range(0, len(a), batch_size)):
        start = i
        end = min(start + batch_size, len(a))
        texts = []
        for j in range(start, end):
            if reverse:
                responses.append(a[j]['response'])
                texts.append(a[j]['response'] + tokenizer.eos_token)
            else:
                contexts.append(a[j]['context'])
                texts.append(a[j]['context'] + tokenizer.eos_token)

        inputs = tokenizer(texts, return_tensors='pt', truncation=True,
                           padding=True, max_length=tokenizer.model_max_length, add_special_tokens=False)
        # input_ids and attention_mask

        for k in inputs:
            inputs[k] = inputs[k].to(device)

        max_length = 100
        output_ids = torch.full(
            [end-start, 1], tokenizer.bos_token_id, dtype=torch.long, device=device)
        past_key_values = None
        ae_past_key_values = None
        bayes_past_key_values = None
        alpha = 0.
        alpha_ts = torch.tensor(alpha, device=device)
        beta = 3.
        beta_ts = torch.tensor(beta, device=device)

        top_p = 0.9
        top_k = 0
        do_sample = True
        unfinished_sents = torch.ones(
            end-start, dtype=torch.long, device=device)
        # print(output_ids)
        bayes_input_ids = torch.full(
            [end-start, 1], bayes_tokenizer.eos_token_id, dtype=torch.long, device=device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        use_direct_generate = False

        if use_direct_generate:
            l = input_ids.size(1)
            # def debug_decorator(func):
            #     def wrapper(*args, **kwargs):
            #         print('Arguments:', args, kwargs)         # Added side-effect
            #         return func(*args, **kwargs)       # Modified return value
            #     return wrapper
            # debug_generate = debug_decorator(model.generate)
            res = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=1, do_sample=do_sample,
                                 top_p=top_p, top_k=top_k, max_length=input_ids.size(1) + max_length)

            output_ids = res[:, l:]

        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            with torch.no_grad():
                for i in range(max_length):
                    # print(inputs)
                    # print(output_ids)
                    # print(f'input ids:{input_ids.size()}, attention_mask:{attention_mask.size()}')
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                    position_ids=position_ids if past_key_values is None else position_ids[:, -1].unsqueeze(
                                        -1),
                                    past_key_values=past_key_values, use_cache=True)
                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values

                    ae_outputs = ae_model(input_ids=input_ids, attention_mask=attention_mask,
                                          position_ids=position_ids if ae_past_key_values is None else position_ids[:, -1].unsqueeze(
                                              -1),
                                          past_key_values=ae_past_key_values, use_cache=True)
                    ae_logits = ae_outputs.logits[:, -1, :]
                    ae_past_key_values = ae_outputs.past_key_values

                    bayes_outputs = bayes_model(
                        input_ids=bayes_input_ids, past_key_values=bayes_past_key_values, use_cache=True)
                    bayes_logits = bayes_outputs.logits[:, -1, :]
                    bayes_past_key_values = bayes_outputs.past_key_values

                    # 正常混合的反向生成
                    logits = top_k_top_p_filtering(logits, top_p=top_p)
                    ensemble_logits = logits * \
                        (1 + alpha_ts + beta_ts) - alpha_ts * \
                        ae_logits - beta_ts * bayes_logits

                    # ensemble_logits = repetition_penalty(output_ids, ensemble_logits, 2.)

                    next_token_logits = top_k_top_p_filtering(
                        ensemble_logits, top_p=top_p, top_k=top_k)

                    # 毒性模型的反向生成
                    # ensemble_logits = ae_logits
                    # next_token_logits = top_k_top_p_filtering(
                    #     ensemble_logits, top_p=top_p, top_k=top_k)

                    # 直接生成context
                    # ensemble_logits = bayes_logits
                    # next_token_logits = top_k_top_p_filtering(
                    #     ensemble_logits, top_p=top_p, top_k=top_k)

                    probs = torch.softmax(next_token_logits, dim=-1)
                    if do_sample:
                        next_tokens = torch.multinomial(
                            probs, num_samples=1).squeeze(1)
                    else:
                        next_tokens = torch.argmax(probs, dim=-1)

                    tokens_to_add = next_tokens * unfinished_sents + \
                        tokenizer.pad_token_id * (1 - unfinished_sents)
                    eos_in_sents = tokens_to_add == tokenizer.eos_token_id
                    unfinished_sents.mul_((~eos_in_sents).long())

                    if unfinished_sents.max() == 0:
                        break

                    output_ids = torch.cat(
                        [output_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                    input_ids = tokens_to_add.unsqueeze(-1)
                    bayes_input_ids = tokens_to_add.unsqueeze(-1)
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))], dim=-1)
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)

        # reply_ids = model.generate(
        #     **inputs, num_beams=1, do_sample=True, top_k=30, max_length=100)
        # print(model.training)
        # outtext = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)
        outtext = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        print(outtext)
        if reverse:
            for text in outtext:
                contexts.append(text)
        else:
            for text in outtext:
                responses.append(text)
        # for j in range(start, end):
            # a[j]['context'] = outtext[j - start]

    df = pd.DataFrame({'context': contexts, 'response': responses})

    with open(outpath, 'w') as outf:
        df.to_csv(outf)


if __name__ == '__main__':
    seed_everything(2022)
    parser = ArgumentParser()
    parser.add_argument(
        '--inpath', type=str, default='../dialogpt_data/prefixtune/test_highInduce.csv')
    parser.add_argument('--outpath', type=str,
                        default='../dialogpt_data/prefixtune/test_highInduce_DexpertDialoGPTMediumReverseBayes_gen.csv')
    parser.add_argument('--ckptpath', type=str,
                        default='../logs/dialogpt_prompttune/1_usePromptFalse_fixlmFalse_promptinitkind_rand_lr2e-05.ckpt')
    parser.add_argument('--ae_ckptpath', type=str,
                        default='../logs/dialogpt_prompttune/1_usePromptFalse_fixlmFalse_promptinitkind_rand_lr2e-05_highInduce_hightoxic.ckpt', help='anti-expert ckptpath')
    # parser.add_argument('--bayes_ckptpath', type=str,
    #                     default='../logs/dialogpt_prompttune/1_usePromptFalse_fixlmFalse_promptinitkind_rand_lr2e-05_onlyresponseTrue_highInduce.ckpt', help='bayes ckptpath')
    parser.add_argument('--bayes_ckptpath', type=str,
                    default='../logs/dialogpt_prompttune/1_usePromptFalse_fixlmFalse_promptinitkind_rand_lr2e-05.ckpt', help='bayes ckptpath')

    parser.add_argument('--resp_sample_num', type=int, default=1)
    parser.add_argument('--reverse', type=bool, default=True)

    args = parser.parse_args()

    dexpert_gen_bayes(args.inpath, args.outpath, ckpt_path=args.ckptpath,
                      ae_ckpt_path=args.ae_ckptpath, bayes_ckpt_path=args.bayes_ckptpath, reverse=args.reverse, resp_sample_num=args.resp_sample_num)

# gen('../blender_data/parlai_adversarial_fewshot_gen_out.json', '../data/huggingface_blender_90M_fewshot_reversed_blender_400M_gen.csv')
# gen('../data/parlai_adversarial.csv', '../data/parlai_original_context_blender_400M_gen.csv', in_is_json=False)
# gen('../blender_data/test_complete.json', '../data/test_complete_originalPost_Blender90M_gen_NegTrain_Blender90M_gen.csv')

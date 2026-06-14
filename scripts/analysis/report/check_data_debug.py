import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from omegaconf import OmegaConf
import sys
sys.path.insert(0, 'src')
from data.utils import preprocess_chat_instance

m = OmegaConf.load('configs/model/Llama-3.2-1B-Instruct.yaml')
model_name = m.model_args.pretrained_model_name_or_path
print('model_name', model_name, flush=True)

tok = AutoTokenizer.from_pretrained(model_name)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print('loading_dataset', flush=True)
ds = load_dataset('locuslab/TOFU', name='full', split='train')
print('dataset_len', len(ds), flush=True)
print('columns', ds.column_names, flush=True)

idxs = np.linspace(0, len(ds)-1, 200, dtype=int)
label_counts=[]
input_lens=[]
for i in idxs:
    row = ds[int(i)]
    out = preprocess_chat_instance(tok, m.template_args, [row['question']], [row['answer']], 512, False)
    labels = out['labels'].numpy()
    input_lens.append(len(out['input_ids']))
    label_counts.append(int((labels != -100).sum()))

arr=np.array(label_counts)
print('avg_input_len', float(np.mean(input_lens)), flush=True)
print('avg_supervised_tokens', float(np.mean(arr)), flush=True)
print('min_supervised_tokens', int(np.min(arr)), flush=True)
print('pct_zero_supervised', float(np.mean(arr==0)), flush=True)
print('pct_supervised_under_8', float(np.mean(arr<8)), flush=True)
print('pct_supervised_under_16', float(np.mean(arr<16)), flush=True)

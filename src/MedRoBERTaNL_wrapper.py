from try_utils import *
from constants import *
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
import torch

#:: all roberta
tokenizer = AutoTokenizer.from_pretrained("CLTL/MedRoBERTa.nl")
model_RobertaForMaskedLM = AutoModelForMaskedLM.from_pretrained("CLTL/MedRoBERTa.nl")

import random
import numpy as np
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

logger = None

class Attention(torch.nn.Module):
    """
    Used when use_attention_weighting = True by get_embedding_for_text
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        attention_socres = self.linear(hidden_states)
        attention_socres = attention_socres.transpose(1, 2)
        attention_weights = torch.nn.functional.softmax(attention_socres, dim=-1)
        context_vector = torch.einsum("bij,bjk->bk", attention_weights, hidden_states)
        return context_vector



def get_embedding_for_text(in_text, use_attention_weighting=F, return_tensors="pt", per_word = F, words_wl = []):
    texts = _batch_roberta_text(in_text)
    reses = []
    res = None
    
    for text in texts:
        is_empty = len(text) < 12
        text = text[3:-4] # strip leading <s> and trailing </s> special tokens
        if is_empty:
            text = f" {text} {text} {text} {text} {text} niet leeg tekst is hier"

        all_embeddings = None
        attentions = None
        with  torch.no_grad():
            excp_counter = 0
            outputs = None
            while outputs is None and len(text) > 2 and excp_counter <50:
                inputs = tokenizer(text, return_tensors="pt")#, truncation=T, max_length=512) # try 1024 , 2048 , tried didnt work..
                inputs['input_ids'].shape
                inputs["output_hidden_states"] = T
                inputs["output_attentions"] = T
                try:
                    outputs = model_RobertaForMaskedLM(**inputs)
                    
                except:
                    excp_counter += 1
                    text = text[1:-1]
                    logger(f"WARN: model_RobertaForMaskedLM failed ({excp_counter}) time(s)")
            all_embeddings = outputs.hidden_states
            
        last_hidden_states = all_embeddings[-1]
       
        if per_word:
            words = text.split()
            masked_wis  = [ 1 if w in words_wl else 0 for w in words ]
            nz_idxs = [i for i,v in enumerate(masked_wis) if v == 1]

            if len(nz_idxs) == 0:
                return torch.zeros(len(words), 768)

            token_ids = inputs['input_ids'][0][1:-1].tolist()
            tokens_lens = [len(tokenizer.decode(x).strip()) for x in token_ids]
            token_idxs_per_word = [[]]*len(words)
            c_wi = 0
            c_w = words[c_wi]
            is_wl = masked_wis[c_wi] == 1
            c_mwi = -1
            if is_wl:
                c_mwi = 0

            n_tokens = len(token_ids)
            for ti, tl in enumerate(tokens_lens):
                c_w = c_w[tl:]
                if is_wl:
                    token_idxs_per_word[c_wi] = token_idxs_per_word[c_wi] + [ti]
                 
                if c_w == '' and ti != n_tokens-1:
                    c_wi +=1
                    c_w = words[c_wi]
                    is_wl = masked_wis[c_wi] == 1
                    if is_wl:
                        c_mwi +=1

            token_embeddings = last_hidden_states[inputs['attention_mask'] == 1]
            for token_idxs in token_idxs_per_word:
                
                word_emb = torch.mean(last_hidden_states[:, token_idxs,], dim=1) if token_idxs != [] else torch.zeros(1, 768)
                if res is None:
                    res = word_emb
                else:
                    res = torch.cat([res, word_emb], dim=0)


        else:
            res = torch.mean(last_hidden_states, dim=1)

        if use_attention_weighting:
            attention_weights = outputs.attentions[-1]
            weighted_embeddings = torch.einsum("nih,bhj->bhj", attention_weights[0], last_hidden_states)
            attention = Attention(last_hidden_states.shape[-1])
            context_vector = attention(weighted_embeddings)
            context_vector = context_vector.detach()
            res = context_vector
        if return_tensors == "np":
            res = res.numpy()
        if is_empty:
            res[:] = 0
        reses += [res]
    if return_tensors == "np":
        res = np.sum(reses, axis=0)
    else:
        res = torch.sum(torch.stack(reses), dim=0)
    return res

def RobertaForMaskedLM_cos_sim_texts(txt1, txt2):
    e1 = get_embedding_for_text(txt1)
    e2 = get_embedding_for_text(txt2)
    return torch.nn.functional.cosine_similarity(e1, e2)

# pass on custom model to top2vec
# need to create a variable embedding_model which is a function
# when called the function should return the embedding of its given documents
def RobertaForMaskedLM_embedding_model(logger1, in_texts, per_word = F, words_wl=[]):
    global logger
    logger = logger1
    embeddings = []
    for text in in_texts:
        embeddings += [get_embedding_for_text(text, return_tensors = "np", per_word = per_word, words_wl=words_wl)[0]]
    return np.array(embeddings)

def _batch_roberta_text(text):
    batched_token_ids = tokenizer(text, truncation=T, max_length=512, return_overflowing_tokens=T, stride=3).input_ids 
    return tokenizer.batch_decode(batched_token_ids)

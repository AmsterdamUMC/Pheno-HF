from try_utils import *
from constants import *

IS_DEBUG = parse_commandline_args(verbose=True)["IS_DEBUG"]
SUBSAMPLE_DATA = parse_commandline_args()["SUBSAMPLE_DATA"]
subsampled_str = "" if not SUBSAMPLE_DATA else "_SUBSAMPLED"

check_if_debugging(IS_DEBUG)

outfile_infix = f"{subsampled_str}"
logfile = f'{os.path.basename(__file__)[:-3]}_{outfile_infix}.log'
logger = get_logger_fn(logfile)
logger(f"Starting ...")

from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
import torch


#::roberta
tokenizer = AutoTokenizer.from_pretrained("CLTL/MedRoBERTa.nl")
model_RobertaForMaskedLM = AutoModelForMaskedLM.from_pretrained("CLTL/MedRoBERTa.nl")
class Attention(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        attention_socres = self.linear(hidden_states)
        attention_socres = attention_socres.transpose(1, 2)
        attention_weights = torch.nn.functional.softmax(attention_socres, dim=-1)
        context_vector = torch.einsum("bij,bjk->bk", attention_weights, hidden_states)
        return context_vector


def _find_max_num_of_batches_per_texts(texts):
    max_n_batches = 0
    for text in texts:
        if len(text) < 512:
            continue
        batched_token_ids = tokenizer(text, truncation=T, max_length=512, return_overflowing_tokens=T, stride=3).input_ids
        n_batches = len(batched_token_ids)
        max_n_batches= max(max_n_batches, n_batches)    
    return max_n_batches

def _batch_roberta_text(text):
    batched_token_ids = tokenizer(text, truncation=T, max_length=512, return_overflowing_tokens=T, stride=3).input_ids
    return tokenizer.batch_decode(batched_token_ids)


def get_embedding_for_text(in_text, use_attention_weighting=F, return_tensors="pt"):
    texts = _batch_roberta_text(in_text)
    reses = []
    res = None
    for text in texts:
        text = text[3:-4] # strip leading <s> and trailing </s> special tokens
        inputs = tokenizer(text, return_tensors="pt")#, truncation=T, max_length=512)
        inputs['input_ids'].shape
        inputs["output_hidden_states"] = T
        inputs["output_attentions"] = T

        all_embeddings = None
        attentions = None
        with  torch.no_grad():
            outputs = model_RobertaForMaskedLM(**inputs)
            all_embeddings = outputs.hidden_states
            attentions =  outputs.attentions
        last_hidden_states = all_embeddings[-1]
        sentence_embedding = torch.mean(last_hidden_states, dim=1)
        res = sentence_embedding 
        if use_attention_weighting:
            attention_weights = attentions[-1]
            weighted_embeddings = torch.einsum("nih,bhj->bhj", attention_weights[0], last_hidden_states)
            attention = Attention(last_hidden_states.shape[-1])
            context_vector = attention(weighted_embeddings)
            context_vector = context_vector.detach()
            res = context_vector
        if return_tensors == "np":
            res = res.numpy()
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

#:: sbert 
model_sbert = SentenceTransformer("NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers")
def sbert_cos_sim_texts(txt1, txt2):
    e1 = model_sbert.encode(txt1, convert_to_tensor=T)
    e1 = e1.unsqueeze(0)
    e2 = model_sbert.encode(txt2, convert_to_tensor=T)
    e2 = e2.unsqueeze(0)
    return torch.nn.functional.cosine_similarity(e1, e2)


#::generics
def cos_sim_texts(txt1, txt2, model_type = "RobertaForMaskedLM"):
    res = None
    if model_type == "RobertaForMaskedLM":
        res = RobertaForMaskedLM_cos_sim_texts(txt1, txt2)
    if model_type == "sbert":
        res = sbert_cos_sim_texts(txt1, txt2)
    return res.item()

HF_texts = ["14 dgn hart onregelmatiger, wordt s nachts met <PERSOON-1> wakker, geen POB, geen dyspnoe, om de 5 min dan onregelmatig, vooral na sporten/zwemmen s nachts veel last, voorheen sporadisch. Vader pacemaker en hartfalen",
            "ER Dhr heeft afgelopen wkn 20 mg furosemide geslikt ipv 40 mg Kwam er recent achter. Klachten; wat dikkere benen bdz en kortademigheid bij inspanning Over 1 week afspraak cardioloog",
            "kennismaking; opgegroeid in schippersgezin <LOCATIE-1>, later met ouders ivm bagger naar <LOCATIE-2>, zelf kraanmachinist geweest bij gemeente; loopt bij cardioloog ivm 3 slechte hartkleppen, kan niet geopereerd worden,, hartfalen, en dieetvoeding, ivm gewichts verlies, nu gewicht redelijke stabiel, echtere houdt niet van dieetsoepen"]

non_HF_texts = ["social call. x mammo en uroloog goed. maar gaat verder zeer moeizaam. alsof uwi en ab reserve verder hebben afgekalfd. gisteren was gedaan bijvoorbeeld, dan hele toer, en <PERSOON-1> heel veel energie. moet niet zo lang meer duren. gaat nog wel naar pianorecital. zorg niet altijd optimaal, vaste kracht nog maar enns per 14 dgn. anderen te snel. leesclub destijds ook al teveel energie",
                "VS MDL geeft aan, 8 poliepen verwijderd 3 jaar terug, van poliep 10-15 jaar tot ontwikkeling kankercellen, comorbiditeit maakt dat het niet opweegt. leeft wrsch niet zo lang. bovendien 2 dgn laxeren ter voorbereiding, ct niet mogelijk dat zegt niets. scopie kan alleen zonder sedatie als ze toch wil",
                "op voorhoof goreined ding. elders ook licht jeukende verheven plakken"]

res = {}
for model_type in ["RobertaForMaskedLM", "sbert"]:
    alike_css = []
    non_alike_css = []
    # compare alike sentences
    for txts in [HF_texts, non_HF_texts]:
        for i,_ in enumerate(txts[1:]):
            cs = cos_sim_texts(txts[i-1], txts[i], model_type = model_type)
            alike_css+= [cs]

    # compare non-alike 
    for i,_ in enumerate(HF_texts):
        for j,_ in enumerate(non_HF_texts):
            cs = cos_sim_texts(HF_texts[i], non_HF_texts[j], model_type = model_type)
            non_alike_css += [cs]

    res[model_type] = {"alike_css" : alike_css, "non_alike_css": non_alike_css}

# the differences are so close, hard to say which one will perform better...
# My gut tells me to use medroberta
np.mean(res["RobertaForMaskedLM"]["alike_css"])
np.mean(res["RobertaForMaskedLM"]["non_alike_css"])

np.mean(res["sbert"]["alike_css"])
np.mean(res["sbert"]["non_alike_css"])


# pass on custom model to top2vec
# need to create a variable embedding_model which is a function
# when called the function should return the embedding of its given documents
def RobertaForMaskedLM_embedding_model(in_texts):
    embeddings = []
    for text in in_texts:
        embeddings += [get_embedding_for_text(text, return_tensors = "np")[0]]
    return np.array(embeddings)

# a_long_text = " ".join([f"alo{i}" for i in range(601)])
# model_sbert.encode(a_long_text, convert_to_tensor=T)
# xxx =RobertaForMaskedLM_embedding_model([a_long_text])



tmp = try_read_pickle(f'pats_dict_quantized_text_SUBSAMPLED.pkl')
text_cols = tmp['text_cols']
x = tmp['x']
del tmp
max_batches_dict_str = '{ '
for txt_col in text_cols:
    txts = [v[txt_col] for v in x.values()]
    t0 = logger(txt_col)

    [get_embedding_for_text(txt, return_tensors="np") for txt in txts]
    t0 = logger(txt_col, t0)
    print("")
    # max_batches_dict_str += f"\n\t\t'{txt_col}' : {_find_max_num_of_batches_per_texts(txts)},"

max_batches_dict_str = max_batches_dict_str[:-1] + '\n}'
logger(max_batches_dict_str)
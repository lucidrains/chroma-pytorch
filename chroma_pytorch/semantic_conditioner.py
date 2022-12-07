import torch
import os
import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging
from tf_bind_transformer.cache_utils import cache_fn, run_once

logging.set_verbosity_error()

def exists(val):
    return val is not None

def map_values(fn, dictionary):
    return {k: fn(v) for k, v in dictionary.items()}

CONTEXT_EMBED_USE_CPU = os.getenv('CONTEXT_EMBED_USE_CPU', None) is not None

if CONTEXT_EMBED_USE_CPU:
    print('calculating context embed only on cpu')

MODELS = dict(
    pubmed = dict(
        dim = 768,
        path = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    )
)

GLOBAL_VARIABLES = dict(model = None, tokenizer = None)

def get_contextual_dim(model_name):
    assert model_name in MODELS
    return MODELS[model_name]['dim']

@run_once('init_transformer')
def init_transformer(model_name):
    path = MODELS[model_name]['path']
    GLOBAL_VARIABLES['tokenizer'] = AutoTokenizer.from_pretrained(path)

    model = AutoModelForMaskedLM.from_pretrained(path)

    if not CONTEXT_EMBED_USE_CPU:
        model = model.cuda()

    GLOBAL_VARIABLES['model'] = model

@torch.no_grad()
def tokenize_text(
    text,
    max_length = 256,
    model_name = 'pubmed',
    hidden_state_index = -1,
    return_cls_token = True
):
    init_transformer(model_name)

    model = GLOBAL_VARIABLES['model']
    tokenizer = GLOBAL_VARIABLES['tokenizer']

    encoding = tokenizer.batch_encode_plus(
        [text],
        add_special_tokens = True,
        padding = True,
        truncation = True,
        max_length = max_length,
        return_attention_mask = True,
        return_tensors = 'pt'
    )

    if not CONTEXT_EMBED_USE_CPU:
        encoding = map_values(lambda t: t.cuda(), encoding)

    model.eval()
    with torch.no_grad():
        outputs = model(**encoding, output_hidden_states = True)

    hidden_state = outputs.hidden_states[hidden_state_index][0]

    if return_cls_token:
        return hidden_state[0]

    return hidden_state.mean(dim = 0)

def get_text_repr(
    texts,
    *,
    device,
    max_length = 256,
    model_name = 'pubmed',
    hidden_state_index = -1,
    return_cls_token = True,
):
    assert model_name in MODELS, f'{model_name} not found in available text transformers to use'

    if isinstance(texts, str):
        texts = [texts]

    get_context_repr_fn = cache_fn(tokenize_text, path = f'contexts/{model_name}')

    representations = [get_context_repr_fn(text, max_length = max_length, model_name = model_name, hidden_state_index = hidden_state_index, return_cls_token = return_cls_token) for text in texts]

    return torch.stack(representations).to(device)

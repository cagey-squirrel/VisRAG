from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from PIL import Image
import os
import json
import numpy as np
import pickle 


def weighted_mean_pooling(hidden, attention_mask):
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps


def save_as_pkl(file_path, object):
    with open(file_path, "wb") as file:
        pickle.dump(object, file)


def read_pkl(file_path):
    with open(file_path, "rb") as file:
        object = pickle.load(file)
    return object


@torch.no_grad()
def encode(model, inputs):
    
    outputs = model(**inputs)
    attention_mask = outputs.attention_mask
    hidden = outputs.last_hidden_state

    reps = weighted_mean_pooling(hidden, attention_mask)   
    embeddings = F.normalize(reps, p=2, dim=1)

    return embeddings


def get_dict_from_json(json_path):
    file = open(json_path, 'r')
    dictionary = json.load(file)
    return dictionary


def batch_embed(input_data, model, tokenizer, step=100):
    start = 0
    batch_list = []
    while start < len(input_data):
        
        end = start + step
        end = min(end, len(input_data))

        print(f'Embedding part {start // step}/{len(input_data) // step}')
        batch = input_data[start:end]
        batch_inputs = {
            'text': batch,
            'image': [None] * len(batch),
            'tokenizer': tokenizer
        }

        embeddings_corpus_batch = encode(model, batch_inputs)
        start = end
        batch_list.append(embeddings_corpus_batch)

    embeddings = torch.cat(batch_list, dim=0)
    return embeddings


def calculate_scores(path_to_top10, queries_path, q_and_a_path):

    chart_qa_id_to_text = get_dict_from_json(queries_path)
    ids = list(chart_qa_id_to_text.keys())
    corpus = chart_qa_id_to_text.values()

    qs_and_as = get_dict_from_json(q_and_a_path)
    queries = [q_and_a['query'] for q_and_a in qs_and_as]
    answers = [q_and_a['label'] for q_and_a in qs_and_as]
    docids = [q_and_a['imgname'].split('.png')[0] for q_and_a in qs_and_as]
    docids = [ids.index(did) for did in docids]

    with open(path_to_top10, "rb") as file:
        top10_indices = pickle.load(file)

    mrr = 0.0
    total_found = 0

    for indices, docid in zip(top10_indices, docids):
        found = 0
        for i, index in enumerate(indices):
            if index == docid:
                mrr += 1 / (i + 1.0)
                found = 1
                break
        total_found += found
    mrr /= len(docids)
    recal = total_found / len(docids)
    print(f'MRR@10 =  {mrr}')
    print(f'recal@10 =  {recal}')


def main():
    model_name_or_path = "openbmb/VisRAG"
    queries_path = r'/home/dzi/VisRAG.git/VisRAG/datasets/chart_qa/chart_qa_ids_to_text.json'
    q_and_a_path = r'/home/dzi/VisRAG.git/VisRAG/datasets/chart_qa/test_augmented.json'
    save_path = "chart_qa_top10_ind_VisRag_1inst_text_and_values_augmented.pkl"

    if os.path.exists(save_path):
        print(f'Error {save_path} already exists')
        exit(1)

    chart_qa_id_to_text = get_dict_from_json(queries_path)
    ids = list(chart_qa_id_to_text.keys())
    corpus = list(chart_qa_id_to_text.values())
    corpus = [c if len(c) else "No data" for c in corpus]

    qs_and_as = get_dict_from_json(q_and_a_path)
    queries = [q_and_a['query'] for q_and_a in qs_and_as]
    answers = [q_and_a['label'] for q_and_a in qs_and_as]
    docid = [q_and_a['imgname'].split('.png')[0] for q_and_a in qs_and_as]
    docid = [ids.index(did) for did in docid]


    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_or_path,torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()

    INSTRUCTION_QUERIES = "Represent this query for retrieving relevant documents: "
    queries = [INSTRUCTION_QUERIES + query for query in queries]
    
    print(f'Embedding corpus')
    embeddings_corpus = batch_embed(corpus, model, tokenizer)
    print(f'Embedding queries')
    embeddings_query = batch_embed(queries, model, tokenizer)
    
    
    scores = (embeddings_query @ embeddings_corpus.T)
    topk = 10
    topk_scores, topk_indices = torch.topk(scores, topk, dim=1)

    topk_scores = topk_scores.detach().cpu().numpy()
    topk_indices = topk_indices.detach().cpu().numpy()

    save_as_pkl(save_path, topk_indices)

    print(f'Ended')

    


if __name__ == "__main__":
    queries_path = r'/home/dzi/VisRAG.git/VisRAG/datasets/chart_qa/chart_qa_ids_to_text_and_values.json'
    q_and_a_path = r'/home/dzi/VisRAG.git/VisRAG/datasets/chart_qa/test_augmented.json'
    path_to_top10 = r"/home/dzi/VisRAG.git/VisRAG/datasets/chart_qa/chartqa_top10_ind_VisRag_1instruct.pkl"
    path_to_top10 = r"chart_qa_top10_ind_VisRag_1inst_text_and_values.pkl"
    path_to_top10 = r"/home/dzi/VisRAG.git/VisRAG/chart_qa_top10_ind_VisRag_1inst_text_and_values_augmented.pkl"
    calculate_scores(path_to_top10, queries_path, q_and_a_path)
    #main()
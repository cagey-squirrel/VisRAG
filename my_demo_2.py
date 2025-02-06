from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from PIL import Image
import os
import json
import numpy as np
import pickle 
from collections import defaultdict
from util import *
condition = False
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, safe_sparse_dot
from time import time

if condition:
    from FlagEmbedding import FlagModel


def get_data_from_json(json_path):
    file = open(json_path, 'r')
    data = json.load(file)
    return data


def batch_embed(input_data, model, embed_query=False, step=100):
    start = 0
    batch_list = []
    while start < len(input_data):
        
        end = start + step
        end = min(end, len(input_data))

        print(f'Embedding part {start // step}/{len(input_data) // step}')
        batch = input_data[start:end]
        if embed_query:
            embeddings = model.encode_queries(batch)
        else:
            embeddings = model.encode(batch)
        start = end
        batch_list.append(embeddings)

    embeddings = np.concatenate(batch_list, axis=0)
    return embeddings


def calculate_metrics_slide(top10_indices_path, qs_as_docids, chart_qa_id_to_text):
    
    ids = list(chart_qa_id_to_text.keys())
    ids = [id for id in ids]

    for id in ids:
        if id[:len("remembering")] == "remembering":
            print(f'found {id}')
    docids = [q_and_a['docid'] for q_and_a in qs_as_docids]
    questions = [q_and_a['question'] for q_and_a in qs_as_docids]
    docids_old = docids.copy()


    all_ids = []
    for did_list in docids:
        id_list = []
        for did in did_list:
            #print(did)
            if did in ids:
                ind = ids.index(did)
            else:
                print(did)
                ind = -1
            id_list.append(ind)
        all_ids.append(id_list)
    docids = all_ids
    
    with open(top10_indices_path, "rb") as file:
        #print(f'top10_indices_path = {top10_indices_path}')
        top10_indices = pickle.load(file)

    mrr = 0.0
    total_found = 0
    for indices, docid, d, q in zip(top10_indices, docids, docids_old, questions):
        found = 0
        for i, index in enumerate(indices):
            if index in docid:
                mrr += 1 / (i + 1.0)
                found = 1
                break

       
        total_found += found
        #if not found:
        #    print(f'Not found for {q}:\n{d}\n')

    mrr /= len(docids)
    recal = total_found / len(docids)
    print(f'MRR@10 =  {mrr}')
    print(f'recal@10 =  {recal}')

    print(f'stagod')
    

def calculate_metrics(top10_indices_path, qs_as_docids, chart_qa_id_to_text):
    
    ids = list(chart_qa_id_to_text.keys())
    #ids = [id.split(".jpeg")[0] for id in ids]
    ids = [id.split(".jpg")[0] for id in ids]
    ids.append('lmfv0228_p9')
    docids = [q_and_a['docid'][0].split(".jpg")[0] for q_and_a in qs_as_docids]
    questions = [q_and_a['question'] for q_and_a in qs_as_docids]
    docids_old = docids.copy()


    #all_ids = []
    #for did_list in docids:
    #    id_list = []
    #    for did in did_list:
    #        ind = ids.index(did)
    #        id_list.append(ind)
    #    all_ids.append(id_list)
    #docids = all_ids
    total = 0
    no_total = 0
    
    docids = [ids.index(did) for did in docids]
    #docids = [ids.index(did) for did_list in docids for did in did_list]
    
    with open(top10_indices_path, "rb") as file:
        #print(f'top10_indices_path = {top10_indices_path}')
        top10_indices = pickle.load(file)

    mrr = 0.0
    total_found = 0
    for indices, docid, d, q in zip(top10_indices, docids, docids_old, questions):
        found = 0
        for i, index in enumerate(indices):
            if type(docid) == list:
                if index in docid:
                    mrr += 1 / (i + 1.0)
                    found = 1
                    break
            else:
                if index == docid:
                    mrr += 1 / (i + 1.0)
                    found = 1
                    break
       
        total_found += found
        #if not found:
        #    print(f'Not found for {q}:\n{d}\n')

    mrr /= len(docids)
    recal = total_found / len(docids)
    print(f'MRR@10 =  {mrr}')
    print(f'recal@10 =  {recal}')

    print(f'stagod')


def testing(qs_as_docids):
    set_of_used_ids = read_pkl("chart_qa_set_of_used_ids.pkl")
    filtered = filter_queries_answers_docs(set_of_used_ids, qs_as_docids)
    save_as_pkl("chart_qa_filtered_qs_as_docids.pkl", filtered)


def filter_queries_answers_docs(set_of_used_ids, qs_as_docids):
    '''
    Filters queries answers and docids
    In original paper they used the subset of the test set
    They filtered the questions for which image with an answer cannot be searched for with RAG
    For example for questions like "Which label has the highest value?"
    it is impossible to know which graph the user is asking about
    '''
    filtered_data = []
    for entry in qs_as_docids:
        docid = entry['docid']
        if docid in set_of_used_ids:
            filtered_data.append(entry)
    return filtered_data


def calculate_question_to_corpus_similarities(chart_qa_id_to_text, qs_and_as, save_path):

    if os.path.exists(save_path):
        print(f'Error {save_path} already exists')
        exit(1)

    corpus = list(chart_qa_id_to_text.values())
    queries = [q_and_a['question'] for q_and_a in qs_and_as]


    model = FlagModel('BAAI/bge-large-en-v1.5', 
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages：",
                  use_fp16=True)

    embeddings_query = batch_embed(queries, model, embed_query=True)
    embeddings_corpus = batch_embed(corpus, model)

    embeddings_query = torch.Tensor(embeddings_query).to("cuda")
    embeddings_corpus = torch.Tensor(embeddings_corpus).to("cuda")

    scores = (embeddings_query @ embeddings_corpus.T)
    topk = 10
    topk_scores, topk_indices = torch.topk(scores, topk, dim=1)

    topk_scores = topk_scores.detach().cpu().numpy()
    topk_indices = topk_indices.detach().cpu().numpy()

    save_as_pkl(save_path, topk_indices)
    
    save_path_for_scores = save_path.split('.pkl')[0]
    save_path_for_scores += "_all_scores.pkl"
    save_as_pkl(save_path_for_scores, scores)


    print(f'Ended')


def filter_used_queries_chart_qa(q_and_a_path):
    qs_and_as = get_data_from_json(q_and_a_path)

def testing2():
    filtered_data_list = read_pkl("/home/dzi/VisRAG.git/VisRAG/datasets/info_vqa/infographics_filtered_data_list.pkl")
    filtered_id_to_text = read_pkl("/home/dzi/VisRAG.git/VisRAG/chart_qa_filtered_ids_filled_empty_to_text_and_values.pkl")

    docids = set()
    for dl in filtered_data_list:
        did = dl['docid']
        docids.add(did)
    
    k = filtered_id_to_text.keys()
    k = set(k)

    print(k.difference(docids))
    print(docids.difference(k))
    print(len(k))
    print(len(docids))


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text


def keyword_search(chart_qa_id_to_text, qs_and_as, save_path):

    text = list(chart_qa_id_to_text.values())
    questions = [q['question'] for q in qs_and_as]


    print(f'Preprocessing corpus')
    start = time()
    # Preprocess documents
    text = [preprocess_text(doc) for doc in text]
    print(f'Preprocessing corpus finished in {time() - start} seconds')

    print(f'Preprocessing queries')
    start = time()
    # Preprocess query
    questions = [preprocess_text(doc) for doc in questions]
    print(f'Preprocessing queries finished in {time() - start} seconds')

    print(f'Encoding corpus')
    start = time()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)
    print(f'Encoding corpus finished in {time() - start} seconds')

    print(f'Encoding queries')
    start = time()
    query_embedding = vectorizer.transform(questions)
    print(f'Encoding queries finished in {time() - start} seconds')

    similarities = cosine_similarity(query_embedding, X)

    save_as_pkl("slide_vqa_sparse.pkl", similarities)


def combine(dense, sparse, path, alpha=0.5):
    combined = (1-alpha) * sparse + alpha * dense

    combined = torch.Tensor(combined)
    topk_scores, topk_indices = torch.topk(combined, 10, dim=1)
    save_as_pkl(path, topk_indices)


    
def main():
    
    
    corpus_path = '/home/dzi/VisRAG.git/VisRAG/datasets/chart_qa/chart_qa_ids_to_text.json'
    q_and_a_path = "chart_qa_filtered_qs_as_docids.pkl"

    corpus_path = '/home/dzi/VisRAG.git/VisRAG/datasets/infovqa/infovqa_id_to_text.pkl'
    q_and_a_path = '/home/dzi/VisRAG.git/VisRAG/datasets/infovqa/infographics_data_list.pkl'

    corpus_path = '/home/dzi/VisRAG.git/VisRAG/datasets/chart_qa/chart_qa_ids_to_text_and_values.pkl'
    q_and_a_path = "/home/dzi/VisRAG.git/VisRAG/chart_qa_qs_as_docd.pkl"

    corpus_path = '/home/dzi/VisRAG.git/VisRAG/chart_qa_filtered_ids_filled_empty_to_text_and_values.pkl'
    q_and_a_path = "/home/dzi/VisRAG.git/VisRAG/chart_qa_qs_as_docd.pkl"

    corpus_path = '/home/dzi/VisRAG.git/VisRAG/infovqa_id_to_gpt_text.pkl'
    q_and_a_path = '/home/dzi/VisRAG.git/VisRAG/datasets/info_vqa/infographics_filtered_data_list.pkl'

    corpus_path = '/home/dzi/VisRAG.git/VisRAG/slide_vqa_image_to_gpt_text.pkl'
    q_and_a_path = '/home/dzi/VisRAG.git/VisRAG/slidevqa_data_list.pkl'

    #corpus_path = '/home/dzi/VisRAG.git/VisRAG/mp_vqa_image_to_gpt_text.pkl'
    #q_and_a_path = '/home/dzi/VisRAG.git/VisRAG/mp_vqa_data_list.pkl'

    save_path = "slide_vqa_bge_gpt_text_all_dense_scores_for_hybrid.pkl"

    # Dictionary which maps image_id to text on that image
    chart_qa_id_to_text = read_pkl(corpus_path)
    #for k in chart_qa_id_to_text:
    #    chart_qa_id_to_text[k] = chart_qa_id_to_text[k].replace("\n", ". ") 

    # This is a list containing a dict for each question
    # Each dict contains at least three fields
    # question, answer, docid
    # docid is the id of the image that contains the answer
    qs_as_docids = read_pkl(q_and_a_path)

    #keyword_search(chart_qa_id_to_text, qs_as_docids, save_path)
    path_to_sparse = "/home/dzi/VisRAG.git/VisRAG/slide_vqa_sparse.pkl"
    path_to_dense = "/home/dzi/VisRAG.git/VisRAG/slide_vqa_bge_gpt_text_all_dense_scores_for_hybrid_all_scores.pkl"
    alpha=0.5
    save_path = "slide_vqa_combined.pkl"

    sparse = read_pkl(path_to_sparse)
    dense = read_pkl(path_to_dense).detach().cpu().numpy()
    combine(dense=dense, sparse=sparse, path=save_path, alpha=alpha)
    calculate_metrics_slide(save_path, qs_as_docids, chart_qa_id_to_text)
    #calculate_question_to_corpus_similarities(chart_qa_id_to_text, qs_as_docids, save_path)
    #calculate_metrics_slide(save_path, qs_as_docids, chart_qa_id_to_text)

if __name__ == "__main__":
    main()
    #testing2()

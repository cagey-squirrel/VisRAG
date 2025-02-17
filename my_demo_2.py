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
condition = True
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


def calculate_metrics(top10_indices_path, qs_as_docids, chart_qa_id_to_text, data):
    
    ids = list(chart_qa_id_to_text.keys())
    ids = [id.split(".jpeg")[0] for id in ids]
    ids = [id.split(".jpg")[0] for id in ids]
    ids = [id.split(".png")[0] for id in ids]
    questions = [q_and_a['question'] for q_and_a in qs_as_docids]
    
    # for mp-dic
    # docids = [q_and_a['docid'][0] for q_and_a in qs_as_docids]
    docids = [q_and_a['docid'] for q_and_a in qs_as_docids]
    
    
    
    #total = 0
    #nottoal = 0
    #for did in docids:
    #    if len(did) == 1:
    #        total += 1
    #    else:
    #        nottoal += 1
    #print(f'total = {total} nototal = {nottoal}')
    names = docids.copy()
    if data == "mp":
        docids = [docid[0] for docid in docids]
        docids = [docid.split(".jpeg")[0] for docid in docids]
        docids = [docid.split(".jpg")[0] for docid in docids]
        docids = [docid.split(".png")[0] for docid in docids]
        ids.append("lmfv0228_p9")
        docids = [ids.index(did) for did in docids]
    elif data == "slide":
        docid_list = []
        for did in docids:
            new_list = []
            for d in did:
                new_list.append(ids.index(d.split('.jpg')[0]))
            docid_list.append(new_list)
        docids = docid_list
    elif data == "plot":
        docid_list = []
        for did in docids:
            new_list = []
            for d in did:
                new_list.append(ids.index(d.split('.png')[0]))
            docid_list.append(new_list)
        docids = docid_list
    else:
        docids = [docid.split(".jpeg")[0] for docid in docids]
        docids = [docid.split(".jpg")[0] for docid in docids]
        docids = [docid.split(".png")[0] for docid in docids]
        docids = [ids.index(did) for did in docids]

    

    #print(f'Len of questions = {len(questions)}')
    
    with open(top10_indices_path, "rb") as file:
        #print(f'top10_indices_path = {top10_indices_path}')
        top10_indices = pickle.load(file)

    mrr = 0.0
    total_found = 0
    non_found_total = 0
    non_found_list = []
    count = 0
    question_to_guesses = {}
    recal_array = []
    for indices, docid, q, name in zip(top10_indices, docids, questions, names):
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
            # MRR @ 1
            #break
        
        guesses = {}
        for i, index in enumerate(indices):
            img_name = ids[index]
            guesses[img_name] =  1 / (i+1)
            #print(f'{img_name}: {score}')
        #print()
        if not found:
            non_found_total += 1
            # Easy find
            #print(f'For question {q} with index {count}, docid {docid} and docid name {name} the document was not found')
            #print(f'Found documets {[names[int(did)] for did in indices]}\n')
            non_found_list.append(count)
        
        question_to_guesses[q] = guesses
        total_found += found
        recal_array.append(found)
        count += 1
    mrr /= len(docids)
    recal = total_found / len(docids)
    print(f'MRR@10 =  {mrr}')
    print(f'recal@10 =  {recal}\n')
    #save_as_pkl(path, question_to_guesses)
    #plt.plot(np.cumsum(recal_array)/len(questions))
    #plt.show()
    #print(question_to_guesses)
    #print(len(non_found))
    #print(non_found_total)
    #print(non_found_list)
    #print(f'len(top10_indices) = {len(top10_indices)}')
    #print(f'len(docids) = {len(docids)}')
    #print(f'len(questions) = {len(questions)}')
    #print(f'len(ids) = {len(ids)}')


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
    

def calculate_metrics_old(top10_indices_path, qs_as_docids, chart_qa_id_to_text):
    
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

    save_as_pkl(save_path, similarities)


def combine(dense, sparse, path, alpha=0.5):
    combined = (1-alpha) * sparse + alpha * dense

    combined = torch.Tensor(combined)
    topk_scores, topk_indices = torch.topk(combined, 10, dim=1)
    save_as_pkl(path, topk_indices)


def calculate_scores(embeddings_path, corpus_path):
    embeddings = read_pkl(embeddings_path)
    corpus = read_pkl(corpus_path)
    scores = embeddings @ corpus.T
    return scores

    
def main(data):
    
    
    corpus_path = '/home/dzi/VisRAG.git/VisRAG/datasets/chart_qa/chart_qa_ids_to_text.json'
    q_and_a_path = "chart_qa_filtered_qs_as_docids.pkl"

    #corpus_path = '/home/dzi/VisRAG.git/VisRAG/datasets/chart_qa/chart_qa_ids_to_text_and_values.pkl'
    #q_and_a_path = "/home/dzi/VisRAG.git/VisRAG/chart_qa_qs_as_docd.pkl"

    #corpus_path = 'datasets/chart_qa/temp_data/chart_qa_filtered_ids_filled_empty_to_text_and_values.pkl'
    #corpus_path = "datasets/chart_qa/chart_qa_new_ids_to_text.pkl"
    #corpus_path = "datasets/chart_qa/chart_qa_brand_new_id_to_text.pkl"
    #corpus_path = "datasets/chart_qa/chart_qa_id_to_gpt_text.pkl"
    #corpus_path = "datasets/chart_qa/chart_qa_new_ids_to_text_filled.pkl"
    #q_and_a_path = "datasets/chart_qa/temp_data/chart_qa_qs_as_docd.pkl"
    save_path = "info_qa_bge_real.pkl"

    corpus_path = '/home/dzi/VisRAG.git/VisRAG/datasets/info_vqa/infovqa_id_to_gpt_text.pkl'
    q_and_a_path = '/home/dzi/VisRAG.git/VisRAG/datasets/info_vqa/infographics_filtered_data_list.pkl'

    #corpus_path = '/home/dzi/VisRAG.git/VisRAG/slide_vqa_image_to_gpt_text.pkl'
    #q_and_a_path = '/home/dzi/VisRAG.git/VisRAG/slidevqa_data_list.pkl'

    #corpus_path = '/home/dzi/VisRAG.git/VisRAG/mp_vqa_image_to_gpt_text.pkl'
    #q_and_a_path = '/home/dzi/VisRAG.git/VisRAG/mp_vqa_data_list.pkl'
    
    id_to_text = read_pkl(corpus_path)
    qs_as_docids = read_pkl(q_and_a_path)

    #for id in id_to_text:
    #    v = id_to_text[id]
    #    old_v = v
    #    v = re.sub(r'sources:.*', '', v, flags=re.IGNORECASE | re.DOTALL)
    #    if "sources:" in old_v.lower():
    #        print(f'sources')
    #    if v != old_v:
    #        print(f'diff: {len(v) - len(old_v)}')
    #        #print(old_v)
    #        #print(v)
    #        print()
    #    id_to_text[id] = v

    print(f'questions: {len(qs_as_docids)}')
    print(f'corpus: {len(id_to_text)}')

    total_len = 0
    for k in id_to_text:
        v = id_to_text[k]
        total_len += len(v)

    print(f'total_len = {total_len}')
    #    chart_qa_id_to_text[k] = v
#
    #exit()
    #calculate_metrics_slide(save_path, qs_as_docids, chart_qa_id_to_text)
    calculate_question_to_corpus_similarities(id_to_text, qs_as_docids, save_path)
    calculate_metrics(save_path, qs_as_docids, id_to_text, data)
    #for alpha in alphas:
    #    combine(dense=dense, sparse=sparse, path=save_path, alpha=alpha)
    #    calculate_metrics(save_path, qs_as_docids, chart_qa_id_to_text, data)


def test_hybrid(data):
    corpus_path = 'datasets/chart_qa/temp_data/chart_qa_filtered_ids_filled_empty_to_text_and_values.pkl'
    corpus_path = "datasets/chart_qa/chart_qa_new_ids_to_text.pkl"
    corpus_path = "datasets/chart_qa/chart_qa_id_to_gpt_text.pkl"
    corpus_path = "datasets/chart_qa/chart_qa_new_ids_to_text_filled.pkl"
    q_and_a_path = "datasets/chart_qa/temp_data/chart_qa_qs_as_docd.pkl"
    path_to_sparse = "chart_qa_sparse.pkl"
    save_path = "chart_qa_bge_new_filled.pkl"

    corpus_path = '/home/dzi/VisRAG.git/VisRAG/datasets/info_vqa/infovqa_id_to_gpt_text.pkl'
    q_and_a_path = '/home/dzi/VisRAG.git/VisRAG/datasets/info_vqa/infographics_filtered_data_list.pkl'
    path_to_sparse = "info_vqa_sparse.pkl"
    save_path = "info_qa_bge_removed_sources_real.pkl"
    save_path = "info_qa_bge_real.pkl"



    save_path_for_scores = save_path.split('.pkl')[0]
    save_path_for_scores += "_all_scores.pkl"
    

    id_to_text = read_pkl(corpus_path)
    qs_as_docids = read_pkl(q_and_a_path)

    for id in id_to_text:
        v = id_to_text[id]
        old_v = v
        v = re.sub(r'sources:.*', '', v, flags=re.IGNORECASE | re.DOTALL)
        if "sources:" in old_v.lower():
            print(f'sources')
        if v != old_v:
            print(f'diff: {len(v) - len(old_v)}')
            #print(old_v)
            #print(v)
            print()
        id_to_text[id] = v

    
    keyword_search(id_to_text, qs_as_docids, path_to_sparse)
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    sparse = read_pkl(path_to_sparse)
    dense = read_pkl(save_path_for_scores).detach().cpu().numpy()


    for alpha in alphas:
        combine(dense=dense, sparse=sparse, path=save_path, alpha=alpha)
        calculate_metrics(save_path, qs_as_docids, id_to_text, data)

if __name__ == "__main__":
    #main('info')
    test_hybrid('info')
    #testing2()

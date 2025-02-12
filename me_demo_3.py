import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from util import *
import numpy as np
from time import time
import os 
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, safe_sparse_dot
from matplotlib import pyplot as plt

def calculate_metrics_slide(top10_indices_path, qs_as_docids, chart_qa_id_to_text):
    
    ids = list(chart_qa_id_to_text.keys())
    ids = [id for id in ids]

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
                #print(did)
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


def calculate_metrics(top10_indices_path, qs_as_docids, chart_qa_id_to_text, data, path):
    
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
    print(str(mrr*100)[:5])
    #print(f'MRR@10 =  {mrr}')
    #print(f'recal@10 =  {recal}\n')
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


def batch_embed(input_data, model, prefix, step=100, max_length=32768):
    start = 0
    batch_list = []
    while start < len(input_data):
        
        end = start + step
        end = min(end, len(input_data))

        start_time = time()
        print(f'Embedding part {start // step}/{len(input_data) // step}', end='')
        batch = input_data[start:end]
        max_length = 32768
        embeddings = model.encode(batch, instruction=prefix, max_length=max_length)
        
        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
        start = end
        batch_list.append(embeddings)

        end_time = time()
        print(f' in {end_time-start_time}s')

    embeddings = torch.cat(batch_list, dim=0)
    return embeddings


def calculate_question_to_corpus_similarities(chart_qa_id_to_text, qs_as_docids, save_path):

    if os.path.exists(save_path):
        print(f'Error {save_path} already exists')
        exit(1)

    # load model with tokenizer
    model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True, device_map="auto").half()

    # Each query needs to be accompanied by an corresponding instruction describing the task.
    task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}
    query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "

    # No instruction needed for retrieval passages
    passage_prefix = ""

    corpus = list(chart_qa_id_to_text.values())
    queries = [q_and_a['question'] for q_and_a in qs_as_docids]

    name = save_path.split('.pkl')[0]

    
    passage_embeddings = batch_embed(corpus, model, passage_prefix, step=1)
    save_as_pkl(f'{name}_just_passage_embeddings.pkl', passage_embeddings)
    exit()

    # get the embeddings
    query_embeddings = batch_embed(queries, model, query_prefix, step=30)
    save_as_pkl(f'{name}_just_query_embeddings.pkl', query_embeddings)

    scores = (query_embeddings @ passage_embeddings.T) * 100

    topk = 10
    topk_scores, topk_indices = torch.topk(scores, topk, dim=1)

    topk_scores = topk_scores.detach().cpu().numpy()
    topk_indices = topk_indices.detach().cpu().numpy()

    save_as_pkl(save_path, topk_indices)


def calculate_scores(embeddings_path, corpus_path):
    embeddings = read_pkl(embeddings_path)
    corpus = read_pkl(corpus_path)
    scores = embeddings @ corpus.T
    return scores



def preprocess_text(text):
    
    # Convert text to lowercase
    #text = text.lower()

    
    # Remove punctuation
    #text = re.sub(r'-', ' ', text) 
    #old_text = text
    #text = re.sub(r'[^\x00-\x7F]', '', text)
    #text = re.sub(r'\b\d+(\.\d+)?%\b|\b[1-9]\d{3}\b', '', text)

    #text = re.sub(r'\(', ' ', text) 
    text = re.sub(r'\bsources:\b.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join(list(set(text.split(' '))))

    # # Replace '-' with space
    return text


def keyword_search(chart_qa_id_to_text, qs_and_as, path, questions_train=None):
    text = list(chart_qa_id_to_text.values())
    questions = [q['question'] for q in qs_and_as]

    # Preprocess documents
    text = [preprocess_text(doc) for doc in text]

    #total =0
    #new_t = []
    #for t in text:
    #    if "sources" in t:#.lower():
    #        total += 1
    #        print(f'Before: {t}')
    #        t = t.split('sources')[0]
    #        print(f'After: {t}')
    #        
    #    new_t.append(t)
    #text = new_t

    #print(f'total = {total}')

    if questions_train:
        questions_train = questions_train[:len(text)]
        questions_train = [preprocess_text(doc) for doc in questions_train]
        training_data = text.copy()
        #training_data.extend(questions_train)
        #questions_train_real = read_pkl('info_questions_real_train.pkl')
        #questions_train_real = [preprocess_text(doc) for doc in questions_train_real]
        #training_data.extend(questions_train_real)
    else:
        training_data = text

    print(f'len train = {len(training_data)}')
    questions = [preprocess_text(doc) for doc in questions]

    vectorizer = TfidfVectorizer()
    vectorizer.fit(training_data)
    X = vectorizer.transform(text)
    
    query_embedding = vectorizer.transform(questions)
    similarities = cosine_similarity(query_embedding, X)
    #fnames = vectorizer.get_feature_names_out() 

    save_as_pkl(path, similarities)


def combine(dense, sparse, path, alpha=0.5):
    combined = (1-alpha) * sparse + alpha * dense

    combined = torch.Tensor(combined)
    topk_scores, topk_indices = torch.topk(combined, 10, dim=1)
    save_as_pkl(path, topk_indices)
    return topk_scores


def test_hybrid(id_to_text, qs_as_docids, path, data='mp'):

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
    
    questions_train = None
    # InvoVQA
    if data == 'info':
        path_to_dense_embeddings = "datasets/info_vqa/temp_files/info_vqa_nvembed_gpt_extractions_2_just_query_embeddings.pkl"
        
        #path_to_dense_corpus = "info_vqa_nvembed_gpt_descriptions_just_passage_embeddings.pkl"
        path_to_dense_corpus = "datasets/info_vqa/temp_files/info_vqa_nvembed_gpt_extractions_just_passage_embeddings.pkl"
        path_to_dense_corpus = "/home/dzi/VisRAG.git/VisRAG/info_vqa_removed_sources_just_passage_embeddings.pkl"
        path_to_sparse = "info_vqa_sparse.pkl"
        save_path = "info_vqa_combined_nvembed.pkl"
        #questions_train = read_pkl('info_questions_train.pkl')
        

    if data == 'mp':
        path_to_dense_embeddings = "datasets/mp_vqa/temp_files/mp_doc_vqa_nvembed_gpt_extractions_just_query_embeddings.pkl"
        path_to_dense_corpus = "datasets/mp_vqa/temp_files/mp_doc_vqa_nvembed_gpt_extractions_just_passage_embeddings.pkl"
        path_to_sparse = "datasets/mp_vqa/temp_files/mp_vqa_sparse.pkl"
        save_path = "datasets/mp_vqa/temp_files/mp_vqa_combined_nvembed.pkl"


    if data == 'chart':
        name = path.split('.pkl')[0]
        path_to_dense_corpus = f'{name}_just_passage_embeddings.pkl'
        path_to_dense_embeddings = f'{name}_just_query_embeddings.pkl'
        path_to_sparse = "datasets/chart_qa/temp_data/chart_qa_sparse.pkl"
        save_path = "datasets/chart_qa/temp_data/chart_qa_combined_nvembed.pkl"
        save_path = "chart_qa_new_order_filled.pkl"


    if data == 'slide':
        path_to_dense_embeddings = "datasets/slide_vqa/temp_data/slide_vqa_nvembed_gpt_extractions_just_query_embeddings.pkl"
        path_to_dense_embeddings = "slide_vqa_nvembed_gpt_text_just_query_embeddings.pkl"
        path_to_dense_corpus = "datasets/slide_vqa/temp_data/slide_vqa_nvembed_gpt_extractions_just_passage_embeddings.pkl"
        path_to_dense_corpus = "slide_vqa_nvembed_gpt_text_just_passage_embeddings.pkl"
        path_to_sparse = "datasets/slide_vqa/temp_data/slide_vqa_sparse.pkl"
        save_path = "datasets/slide_vqa/temp_data/slide_vqa_combined_nvembed.pkl"


    if data == 'plot':
        corpus_path = 'datasets/plot_qa/plot_qa_id_to_gpt_text.pkl'
        q_and_a_path = 'datasets/plot_qa/plot_qa_datalist.pkl'
        path_to_dense_embeddings = 'datasets/plot_qa/temp_files/plot_qa_questions_just_query_embeddings.pkl'
        path_to_dense_corpus = "datasets/plot_qa/temp_files/plotqa_total_corpus_embeddings.pkl"
        path_to_sparse = "plot_qa_sparse.pkl"
        save_path = "plot_qa_combined_nvembed.pkl"

    
    
    keyword_search(id_to_text, qs_as_docids, path_to_sparse, questions_train=questions_train)
    
    dense = calculate_scores(path_to_dense_embeddings, path_to_dense_corpus)
    dense = dense.detach().cpu().numpy()
    sparse = read_pkl(path_to_sparse)

    alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    #alphas = [0.7]
    #alphas = [1]
    save_question_to_guesses_path = f"{data}_question_to_guesses.pkl"
    for alpha in alphas:
        topk_scores = combine(dense=dense, sparse=sparse, path=save_path, alpha=alpha)
        calculate_metrics(save_path, qs_as_docids, id_to_text, data, save_question_to_guesses_path)


def encode_and_metrics(chart_qa_id_to_text, qs_as_docids, save_path, data):
    calculate_question_to_corpus_similarities(chart_qa_id_to_text, qs_as_docids, save_path)
    #calculate_metrics_slide(save_path, qs_as_docids, chart_qa_id_to_text)
    calculate_metrics(save_path, qs_as_docids, chart_qa_id_to_text, data, None)


def main(data):
    if data == "plot":
        corpus_path = "/home/dzi/VisRAG.git/VisRAG/plotqa_0_2500.pkl"
        corpus_path = "/home/dzi/VisRAG.git/VisRAG/plotqa_image_name_to_text_2000_4500.pkl"
        corpus_path = "/home/dzi/VisRAG.git/VisRAG/plotqa_image_name_to_text_4500_7000.pkl"
        corpus_path = "/home/dzi/VisRAG.git/VisRAG/plotqa_image_name_to_text_7000_8000.pkl"
        corpus_path = "/home/dzi/VisRAG.git/VisRAG/plotqa_full_image_name_to_text_8000_9000.pkl"
        corpus_path = "/home/dzi/VisRAG.git/VisRAG/plotqa_image_name_to_text_9500_end.pkl"
        corpus_path = "/home/dzi/VisRAG.git/VisRAG/plotqa_image_name_to_text_9000_9500.pkl"
        save_path = "plot_qa_test.pkl"

        q_and_a_path = "/home/dzi/VisRAG.git/VisRAG/datasets/plot_qa/plot_qa_datalist.pkl"
        save_path = ""

    if data == 'info':
        corpus_path = 'datasets/info_vqa/infovqa_id_to_gpt_text.pkl'
        #corpus_path = 'infovqa_id_to_gpt_decriptions_fixed.pkl'
        q_and_a_path = 'datasets/info_vqa/infographics_filtered_data_list.pkl'
        save_path = "info_vqa_removed_sources.pkl"

    if data == 'slide':
        corpus_path = 'datasets/slide_vqa/slide_vqa_image_to_gpt_text.pkl'
        q_and_a_path = 'datasets/slide_vqa/slide_vqa_data_list.pkl'
        save_path = ""

    if data == 'chart':
        q_and_a_path = "datasets/chart_qa/temp_data/chart_qa_qs_as_docd.pkl"
        corpus_path = 'datasets/chart_qa/temp_data/chart_qa_filtered_ids_filled_empty_to_text_and_values.pkl'
        corpus_path = "datasets/chart_qa/chart_qa_new_ids_to_text.pkl"
        corpus_path = "datasets/chart_qa/chart_qa_new_ids_to_text_filled.pkl"
        save_path = "chart_qa_new_order_filled.pkl"

    if data == 'mp':
        corpus_path = 'datasets/mp_vqa/mp_vqa_image_to_gpt_text.pkl'
        q_and_a_path = 'datasets/mp_vqa/mp_vqa_data_list.pkl'
        save_path = ""

    # Dictionary which maps image_id to text on that image
    id_to_text = read_pkl(corpus_path)
    print(f'len(chart_qa_id_to_text) = {len(id_to_text)}')

    

    
    qs_as_docids = read_pkl(q_and_a_path)
    print(f'len(qs_as_docids) = {len(qs_as_docids)}')

    #encode_and_metrics(id_to_text, qs_as_docids, save_path, data)
    test_hybrid(id_to_text, qs_as_docids, save_path, data)

    # [[87.42693328857422, 0.46283677220344543], [0.965264618396759, 86.03721618652344]]
import json

def get_data_from_json(json_path):
    file = open(json_path, 'r')
    data = json.load(file)
    return data


def testing():
    filtered = read_pkl("chartqa_filtered_questions_text.pkl")
    filtered = set(filtered)
    human = get_data_from_json(f'datasets/chart_qa/test_human.json')
    filtered_data = []
    total = 0
    inside = 0
    for h in human:
        if h['query'] in filtered:
            print(f'YES')
            inside += 1
            total += 1
            d = {}
            d['question'] = h['query']
            d['answer'] = h['label']
            d['docid'] = h['imgname'].split('.png')[0]
            filtered_data.append(d)
            print(h['imgname'])
        else:
            print(f'NO')
            total += 1
    print(f'Inside/Total = {inside}/{total}')
    save_as_pkl("chart_qa_qs_as_docd.pkl", filtered_data)

    #print(filtered)


def testing_2():
    pass


if __name__ == "__main__":
    #testing_2()
    #test_hybrid('info')
    main('info')

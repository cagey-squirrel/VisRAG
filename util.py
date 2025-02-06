import json 
import pickle


def load_json_as_dict(path_to_json):
    file = open(path_to_json, 'r')
    data = json.load(file)

    return data


def save_as_pkl(file_path, object):
    with open(file_path, "wb") as file:
        pickle.dump(object, file)


def read_pkl(file_path):
    with open(file_path, "rb") as file:
        object = pickle.load(file)
    return object
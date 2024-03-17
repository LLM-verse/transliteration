import pickle
import zipfile
import wget
import os 
from keras.layers import TextVectorization
import tensorflow as tf

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def unzip_model(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def save_pickle( dict_to_save , file_path ):        
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(dict_to_save, f)
        print(f"File saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the fle: {e}")


def download_file( url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        wget.download(url, filename)
        print("Download complete.")


def verify_files( lang ):
    
    if not os.path.exists(f"models/{lang}/{lang}_vectorization.pkl"):
        return f"models/{lang}/{lang}_vectorization.pkl is missing"
    
    if not os.path.exists(f"models/{lang}/translit_model.zip"):
        return f"models/{lang}/translit_model.zip is missing"
    
    if not os.path.exists("models/eng_vectorization.pkl"):
        return "models/eng_vectorization.pkl is missing"

    if not os.path.exists("models/{lang}/translit_model"):
        unzip_model(f"models/{lang}/translit_model.zip", f"models/{lang}/")

    return True


def initialize_resources(lang):

    src_config = load_pickle(f"models/{lang}/{lang}_vectorization.pkl")
    src_config['dtype'] = "string"
    src_vectorization = TextVectorization.from_config(src_config)
    
    tgt_config = load_pickle(f"models/eng_vectorization.pkl")
    tgt_config['dtype'] = "string"
    tgt_vectorization = TextVectorization.from_config(tgt_config)
    tgt_vocab = tgt_vectorization.get_vocabulary()
    tgt_index_lookup = dict(zip(range(len(tgt_vocab)), tgt_vocab))
    tgt_index_lookup_vector = tf.constant(list(tgt_index_lookup.values()))

    model = tf.saved_model.load(f"models/{lang}/translit_model")

    return src_vectorization, tgt_index_lookup_vector, model
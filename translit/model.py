import os
import re
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from .util import initialize_resources, load_pickle

class Model():
    def __init__(self, lang="tel", word_batch = 10_000, cached_dict=None):

        self.max_decoded_word_length = 30
        self.compiled_pattern = re.compile(r'[^\u0C00-\u0C7F]+(?<![\u0C00-\u0C7F] )')
        
        self.lang = lang
        self.word_batch = word_batch
        self.identifiers= [" |&&indicc&&1| ", "|&&indicc&&2|", " |&&indicc&&3| "]
        
        self.cached_dict = load_pickle(cached_dict) if cached_dict is not None else {}        
        self.cached_dict[" "] = " "
        self.cached_dict[self.identifiers[0].strip()] = self.identifiers[0].strip()

        self.src_vectorization, self.tgt_index_lookup_vector, self.transformer = initialize_resources(self.lang)
        
    
    def preprocess_text(self, text):
        replacements = []

        def repl_func(match):
            replacements.append( match.group())
            return self.identifiers[0]

        processed_text = self.compiled_pattern.sub(repl_func, text)
        words = processed_text.split(" ")
        if processed_text[-1] == " ":
            words.pop()
            words.append(" ")

        return words, replacements
    
    def postprocess_text(self, processed_text, replacements):
        for replacement in replacements:
            processed_text = processed_text.replace(self.identifiers[0], replacement, 1)
        return processed_text

    @tf.function
    def update_column_by_vector(self, tensor, vector, index):
        vector_casted = tf.cast(vector, tensor.dtype)
        num_rows = tf.shape(tensor)[0]
        indices = tf.stack([tf.range(num_rows), tf.fill([num_rows], index)], axis=1)
        updates = tf.squeeze(vector_casted, axis=-1)  # Removing unnecessary dimensions
        updated_tensor = tf.tensor_scatter_nd_update(tensor, indices, updates)
        return updated_tensor

    @tf.function
    def decode_sentence_vec(self, processed_words):
        
        tokenized_input_words = self.src_vectorization(processed_words)

        tokenized_target_words = tf.zeros_like(tokenized_input_words, dtype=tf.int64)
        first_col_update = tf.fill([tf.shape(processed_words)[0], 1], 2)
        tokenized_target_words = self.update_column_by_vector(tokenized_target_words, first_col_update, 0)

        for i in tf.range(1, self.max_decoded_word_length + 1):
            preds = self.transformer.serve([tokenized_input_words, tokenized_target_words[:, :-1]])
            pred_index = tf.argmax(preds[:, i-1, :], axis=-1)
            pred_index_casted = tf.cast(tf.reshape(pred_index, (-1, 1)), dtype=tf.int64)
            tokenized_target_words = self.update_column_by_vector(tokenized_target_words, pred_index_casted, i)

        gathered_chars = tf.gather(self.tgt_index_lookup_vector, tokenized_target_words, axis=0)
        mask = tf.logical_and(gathered_chars != "", gathered_chars != ">")
        filtered_chars = tf.boolean_mask(gathered_chars, mask)
        joined_word = tf.strings.reduce_join(filtered_chars, axis=-1)
        return joined_word

    def translit(self, data):
        
        Flag = False
        
        if isinstance(data, list):
            Flag = True
            data = self.identifiers[2].join(data)

        words, replacements = self.preprocess_text(data)  
        
        trainslit_words = []
        words_to_translit = []

        for w in words:

            if w in self.cached_dict.keys():
                trainslit_words.append(self.cached_dict[w])
            elif w:
                words_to_translit.append(f"<{w}>")
                trainslit_words.append(self.identifiers[1])

        translit_text = " ".join(trainslit_words)
        
        if words_to_translit:
            if len(words_to_translit) > self.word_batch:
                num_batches = (len(words_to_translit) + self.word_batch - 1) // self.word_batch
                word_batches = [words_to_translit[i*self.word_batch:(i+1)*self.word_batch] for i in range(num_batches)]

                model_outputs = [self.decode_sentence_vec(b).numpy().decode('utf-8')[1:].split("<") for b in word_batches]
                # if model outputs is a list of lists, then we need to flatten it to a single list of strings
                # model outputs can be a list of strings if the input is a single batch
                if isinstance(model_outputs[0], list):
                    model_outputs = [item for sublist in model_outputs for item in sublist]


            else:
                model_outputs = self.decode_sentence_vec(words_to_translit).numpy().decode('utf-8')[1:].split("<")
            
            for i in range(len(words_to_translit)):
                translit_text = translit_text.replace(self.identifiers[1], model_outputs[i], 1)
                self.cached_dict[words_to_translit[i][1:-1]] = model_outputs[i]
            
        translit_text = self.postprocess_text(translit_text, replacements)
        
        if Flag:
            return translit_text.split(self.identifiers[2])
        
        return translit_text

    # def translit(self, data):
        
    #     print("HIII")
    #     words, replacements = self.preprocess_text(data)  
        
    #     trainslit_words = []
    #     words_to_translit = []

    #     for w in words:

    #         if w in self.cached_dict.keys():
    #             trainslit_words.append(self.cached_dict[w])
    #         elif w:
    #             words_to_translit.append(f"<{w}>")
    #             trainslit_words.append(self.identifiers[1])

    #     translit_text = " ".join(trainslit_words)
        
    #     if words_to_translit:
    #         if len(words_to_translit) > self.word_batch:
    #             num_batches = (len(words_to_translit) + self.word_batch - 1) // self.word_batch
    #             word_batches = [words_to_translit[i*self.word_batch:(i+1)*self.word_batch] for i in range(num_batches)]

    #             model_outputs = [self.decode_sentence_vec(b).numpy().decode('utf-8')[1:].split("<") for b in word_batches]

    #         else:
    #             model_outputs = self.decode_sentence_vec(words_to_translit).numpy().decode('utf-8')[1:].split("<")
            
    #         for i in range(len(words_to_translit)):
    #             translit_text = translit_text.replace(self.identifiers[1], model_outputs[i], 1)
    #             # self.cached_dict[words_to_translit[i][1:-1]] = model_outputs[i]
            
    #     translit_text = self.postprocess_text(translit_text, replacements)
        
    #     return translit_text
    

    

    

    

    

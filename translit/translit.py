import concurrent.futures
from .model import Model
from .util import verify_files

class Translit:

    def __init__(self, lang = "tel", word_batch = 10_000, cached_dict=None):

        # self.use_multiprocessing = use_multiprocessing
        # , use_multiprocessing=True
        self.lang = lang
        self.word_batch = word_batch
        self.cached_dict = cached_dict

        verify_files(self.lang)

        self.model = self.make_translit_model()

    def make_translit_model(self):
        return Model(lang=self.lang, word_batch=self.word_batch, cached_dict=self.cached_dict)

    def translit(self, data):
        return self.model.translit(data)

    # def translit_single_item(self, model, data):
    #     return model.translit(data)

    # def translit_batch(self, batch):
    #     translit_model = self.make_translit_model()
    #     batch_translit_texts = []
    #     for item in batch:
    #         batch_translit_texts.append(self.translit_single_item(translit_model, item))
    #     return batch_translit_texts

    # def translit_with_multiprocessing(self, data, batch_size):
    #     # Split data into batches
    #     data_batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
    #     translit_texts = []
        
    #     with concurrent.futures.ProcessPoolExecutor() as executor:
    #         futures = []
    #         for batch in data_batches:
    #             # Create a Translit model for each batch
                
    #             futures.append(executor.submit(self.translit_batch, batch))
            
    #         for future in concurrent.futures.as_completed(futures):
    #             translit_texts.extend(future.result())

    #     return translit_texts

    # def translit_without_multiprocessing(self, data):
    #     translit_model = self.make_translit_model()
    #     return self.translit_batch( data)

    # def translit(self, data, batch_size=100):
    #     if not data:
    #         return []

    #     if isinstance(data, str):
    #         data = [data]  # Convert single string to a list with one element
        
    #     if self.use_multiprocessing:
    #         return self.translit_with_multiprocessing(data, batch_size)
    #     else:
    #         return self.translit_without_multiprocessing(data)


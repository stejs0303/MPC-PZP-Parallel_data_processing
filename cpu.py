from multiprocessing import cpu_count, Pool
from collected_data import Collected_data 
from collections import Counter
from math import ceil
from load import load
import threading
import time

MINIMUM_LETTERS: int = 4
MAXIMUM_LETTERS: int = 8

def process_data(data: list, stop_words: list):   
    filtered_data: Counter = {}
    counter: int = 0
    
    for word in data:
        if len(word) >= MINIMUM_LETTERS and len(word) <= MAXIMUM_LETTERS and word not in stop_words:
            filtered_data[word] = filtered_data.get(word, 0) + 1
            counter += 1
            
    return filtered_data, counter

def single_threaded(data: list, stop_words: list):
  
    start = time.perf_counter()
    
    filtered_data, counter = process_data(data, stop_words)
          
    filtered = time.perf_counter()
          
    most_frequent_word = max(filtered_data, key=filtered_data.get)
    most_frequent_word_count = filtered_data.get(most_frequent_word)
    
    least_frequent_word = min(filtered_data, key=filtered_data.get)
    least_frequent_word_count = filtered_data.get(least_frequent_word)
    
    stop = time.perf_counter()
    
    return Collected_data("CPU", 1, most_frequent_word, 
                          most_frequent_word_count, least_frequent_word, 
                          least_frequent_word_count, counter, 
                          start, start, start, filtered, stop)

def multi_threaded_multiprocessing(data: list, stop_words: list):
    
    start = time.perf_counter()
    
    n_cores = cpu_count()
    step = ceil(len(data)/n_cores)
    data_chunks = []
    
    for idx in range(0, len(data), step):
        data_chunks.append((data[idx:idx+step], stop_words.copy()))
    
    data_prepared = time.perf_counter()
    
    with Pool() as pool:
        results = pool.starmap(process_data, data_chunks, 1)

    data_filtered = time.perf_counter()

    filtered_data: Counter = Counter()
    counter: int = 0
    
    for partial_filtered_data, partial_counter in results:
        filtered_data += partial_filtered_data
        counter += partial_counter   
    
    most_frequent_word = max(filtered_data, key=filtered_data.get)
    most_frequent_word_count = filtered_data.get(most_frequent_word)
    
    least_frequent_word = min(filtered_data, key=filtered_data.get)
    least_frequent_word_count = filtered_data.get(least_frequent_word)
    
    stop = time.perf_counter()
    
    return Collected_data("CPU", n_cores, most_frequent_word, 
                          most_frequent_word_count, least_frequent_word, 
                          least_frequent_word_count, counter, 
                          start, data_prepared, data_prepared, data_filtered, stop)
    
def multi_threaded_threading(data: list, stop_words: list):
    filtered_data: Counter = Counter()
    threads: list = []
    counter: int = 0
    
    start = time.perf_counter()
    
    n_cores = cpu_count()
    step = ceil(len(data)/n_cores)
    
    for idx in range(0, len(data), step):
        thread = CustomThread(data[idx:idx+step], stop_words.copy())
        threads.append(thread)
    
    for thread in threads:
        thread.start()

    data_prepared = time.perf_counter()
    
    for thread in threads:
        thread.join()
        
    for thread in threads:
        filtered_data += thread.filtered_data
        counter += thread.counter
        
    data_filtered = time.perf_counter()
    
    most_frequent_word = max(filtered_data, key=filtered_data.get)
    most_frequent_word_count = filtered_data.get(most_frequent_word)
    
    least_frequent_word = min(filtered_data, key=filtered_data.get)
    least_frequent_word_count = filtered_data.get(least_frequent_word)
    
    stop = time.perf_counter()
    
    return Collected_data("CPU", n_cores, most_frequent_word, 
                          most_frequent_word_count, least_frequent_word, 
                          least_frequent_word_count, counter, 
                          start, data_prepared, data_prepared, data_filtered, stop)

class CustomThread(threading.Thread):
    filtered_data: Counter
    stop_words: list
    counter: int
    data: list
    
    def __init__(self, data: list, stop_words: list):
        threading.Thread.__init__(self)
        self.stop_words = stop_words
        self.filtered_data = Counter()
        self.data = data
        self.counter = 0
        
    def run(self):
        self.filtered_data, self.counter = process_data(self.data, self.stop_words)
            
if __name__=="__main__":
    data, stop_words = load("./files/data.txt", "./files/stop_words.txt")
    print(single_threaded(data, stop_words))
    print(multi_threaded_threading(data, stop_words))
    print(multi_threaded_multiprocessing(data, stop_words))
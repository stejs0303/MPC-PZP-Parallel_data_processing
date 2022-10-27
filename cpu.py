from collected_data import Collected_data 
from multiprocessing import cpu_count
from collections import Counter
from math import ceil
import threading
import time
   
def single_threaded(data: list, stop_words: list):
    filtered_data: dict = {}
    counter: int = 0
    
    start = time.time_ns()
    
    for word in data:
        if len(word) >= 4 and len(word) <= 8 and word not in stop_words:
            filtered_data[word] = filtered_data.get(word, 0) + 1
            counter += 1
            
    most_frequent_word = max(filtered_data, key=filtered_data.get)
    most_frequent_word_count = filtered_data.get(most_frequent_word)
    
    least_frequent_word = min(filtered_data, key=filtered_data.get)
    least_frequent_word_count = filtered_data.get(least_frequent_word)
    
    stop = time.time_ns()
    
    return Collected_data("CPU", 1, most_frequent_word, 
                          most_frequent_word_count, least_frequent_word, 
                          least_frequent_word_count, counter, start, stop)


def multi_threaded(data: list, stop_words: list):
    filtered_data: Counter = Counter()
    threads: list = []
    counter: int = 0
    
    start = time.time_ns()
    
    n_cores = cpu_count()
    step = ceil(len(data)/n_cores)
    for idx in range(0, len(data), step):
        thread = CustomThread(data[idx:idx+step], stop_words.copy())   
        threads.append(thread)
    
    for thread in threads:
        thread.start()
        
    for thread in threads:
        thread.join()
        filtered_data += Counter(thread.filtered_data)
        counter += thread.counter
    
    most_frequent_word = max(filtered_data, key=filtered_data.get)
    most_frequent_word_count = filtered_data.get(most_frequent_word)
    
    least_frequent_word = min(filtered_data, key=filtered_data.get)
    least_frequent_word_count = filtered_data.get(least_frequent_word)
    
    stop = time.time_ns()
    
    return Collected_data("CPU", n_cores, most_frequent_word, 
                          most_frequent_word_count, least_frequent_word, 
                          least_frequent_word_count, counter, start, stop)


class CustomThread(threading.Thread):
    filtered_data: dict
    stop_words: list
    counter: int
    data: list
    
    def __init__(self, data: list, stop_words: list):
        threading.Thread.__init__(self)
        self.stop_words = stop_words
        self.filtered_data = dict()
        self.data = data
        self.counter = 0
        
    def run(self):
        for word in self.data:
            if len(word) >= 4 and len(word) <= 8 and word not in self.stop_words:
                self.filtered_data[word] = self.filtered_data.get(word, 0) + 1
                self.counter += 1
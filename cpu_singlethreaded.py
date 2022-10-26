from data_struct import Collected_data 
import time
   
def single_thread(data: list, stop_words: list):
    filtered_data = dict()
    counter: int = 0
    
    start = time.time_ns()
    
    for word in data:
        if len(word) >= 4 and len(word) <= 8 and word not in stop_words:
            filtered_data[word] = filtered_data.get(word, 0) + 1
            counter += 1
            
    most_frequent = max(filtered_data, key=filtered_data.get)
    most_frequent_count = filtered_data.get(most_frequent)
    
    least_frequent = min(filtered_data, key=filtered_data.get)
    least_frequent_count = filtered_data.get(least_frequent)
    
    stop = time.time_ns()
    
    return Collected_data("CPU-singlethreaded", most_frequent, most_frequent_count, least_frequent, least_frequent_count, counter, start, stop)
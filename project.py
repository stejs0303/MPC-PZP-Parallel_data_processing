#Author:            Jan Stejskal
#School ID:         211272
#Python version:    3.10.7
#Platform:          Windows 11
#Packages:          requirements.txt

from collected_data import Collected_data
from create_graph import show_graph
from load import load
import cpu, gpu

data_path: str = "./files/data.txt"
stop_words_path: str = "./files/stop_words.txt"

def main():
    data, stop_words = load(data_path, stop_words_path)
    
    single_threaded: Collected_data() = cpu.single_threaded(data, stop_words)
    multi_threaded: Collected_data() = cpu.multi_threaded(data, stop_words)
    multi_threaded_gpu: Collected_data() = gpu.multi_threaded(data, stop_words)
    
    '''
    for _ in range(10):  
        single_threaded += cpu.single_threaded(data, stop_words)
        multi_threaded += cpu.multi_threaded(data, stop_words)
        multi_threaded_gpu += gpu.multi_threaded(data, stop_words)
    '''
    
    print(single_threaded)
    print(multi_threaded)
    print(multi_threaded_gpu)
    
    show_graph(single_threaded, multi_threaded, multi_threaded_gpu)
    
if __name__ == "__main__":
    main()
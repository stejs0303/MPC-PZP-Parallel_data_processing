#Author:            Jan Stejskal
#School ID:         211272
#Python version:    3.10.7
#Platform:          Windows 11
#Packages:          requirements.txt

import load as ld
import cpu
import gpu

data_path: str = "./files/data.txt"
stop_words_path: str = "./files/stop_words.txt"

def main():
    data, stop_words = ld.load(data_path, stop_words_path)
    
    single_threaded = cpu.single_threaded(data, stop_words)
    multi_threaded = cpu.multi_threaded(data, stop_words)
    multi_threaded_gpu = gpu.multi_threaded(data, stop_words)
    
    print(single_threaded)
    print(multi_threaded)
    print(multi_threaded_gpu)
    
if __name__ == "__main__":
    main()
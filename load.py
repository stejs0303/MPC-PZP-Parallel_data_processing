import time
import re

def isnt_number(s):
    try:
        float(s)
        return False
    except ValueError:
        return True

def load(data_path: str, stop_words_path: str):
    
    data, stop_words = [], []

    with open(data_path, 'r') as file_data:
        while(True):
            line = file_data.readline()

            if(not line): break

            data.extend([str.lower(word) 
                         for word in re.split(r"_|[\b\W\b]+", line)
                         if (word and isnt_number(word))])
    
    with open(stop_words_path, 'r') as file_stop_words:
        stop_words = [str.lower(word.strip(" \n\r")) 
                      for word in file_stop_words.readlines()]
    
    return data, stop_words

if __name__ == "__main__":
    start = time.time_ns()
    data, stop_words = load("./files/data.txt", "./files/stop_words.txt")
    stop = time.time_ns()
    #print(f"{(stop-start)/int(1e6)} ms")
    #print(data[:50])
    #print(stop_words)
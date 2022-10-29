import time
import re

def load(data_path: str, stop_words_path: str):
    
    data, stop_words = [], []

    with open(data_path, 'r') as file_data:
        while(True):
            line = file_data.readline()

            if(not len(line)): break

            data.extend([word for word in re.split(" |,|;|\n|\r|\.|\\|\*|\+|\?|\[|\]|\(|\)|\{|\}|\!|\:|\-\-|\-\-\-|_|\'|\"", line) if word])
    
    with open(stop_words_path, 'r') as file_stop_words:
        stop_words = [word.strip(" \n\r") for word in file_stop_words.readlines()]
    
    return data, stop_words

if __name__ == "__main__":
    start = time.time_ns()
    load("./files/data.txt", "./files/stop_words.txt")
    stop = time.time_ns()
    print(f"{(stop-start)/int(1e6)} ms")
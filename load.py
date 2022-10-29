import time
import re

def load(data_path: str, stop_words_path: str):
    
    data, stop_words = [], []

    with open(data_path, 'r') as file:
            file_data = file.read()
            data.append([word for word in re.split(" |,|;|\n|\r|\.|\\|\*|\+|\?|\[|\]|\(|\)|\{|\}|\!|\:|\-\-|\-\-\-|_|\'|\"", file_data) if word])
    
    with open(stop_words_path, 'r') as file:
        file_stop_words = file.read()
        stop_words.append([word.strip(" \n\r") for word in file_stop_words])
    
    return data, stop_words

if __name__ == "__main__":
    start = time.time_ns()
    load("./files/data.txt", "./files/stop_words.txt")
    stop = time.time_ns()
    print(f"{(stop-start)/int(1e6)} ms")
import re

def load(data_path: str, stop_words_path: str):
    
    stop_words, data = [], []
    
    with open(stop_words_path, 'r') as file_stop_words:
        stop_words = [word.strip(" \n\r") for word in file_stop_words.readlines()]

    with open(data_path, 'r') as file_data:
        while(True):
            line = file_data.readline()

            if(not len(line)): break

            data.extend([word for word in re.split(" |,|;|\n|\r|\.|\\|\*|\+|\?|\[|\]|\(|\)|\{|\}|\!|\:|\-\-|\-\-\-|_|\'|\"", line) if word ])
    
    return data, stop_words
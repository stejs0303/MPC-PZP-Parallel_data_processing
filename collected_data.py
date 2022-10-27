from math import inf

class Collected_data:
    type: str
    threads: int
    
    most_frequent_word: str
    most_frequent_word_count: int
    
    least_frequent_word: str
    least_frequent_word_count: int
    
    final_word_count: int
    
    start_time: str
    end_time: str
    
    
    def __init__(self, type:str = "", threads: int = 0,
                 most_frequent_word: str = "", most_frequent_word_count: int = 0, 
                 least_frequent_word: str = "", least_frequent_word_count: int = inf, 
                 final_word_count: int = 0, start_time: int = 0, end_time: int = 0):
        
        self.type = type
        self.threads = threads
        self.most_frequent_word = most_frequent_word
        self.most_frequent_word_count = most_frequent_word_count
        self.least_frequent_word = least_frequent_word
        self.least_frequent_word_count = least_frequent_word_count
        self.final_word_count = final_word_count
        self.start_time = start_time
        self.end_time = end_time

    def __repr__(self) -> str:
        string = f'''
        Processed on: {self.type}, number of threads: {self.threads}.
        Most frequent word: \"{self.most_frequent_word}\", appeared: {self.most_frequent_word_count} in the text.
        Least frequent word: \"{self.least_frequent_word}\", appeared: {self.least_frequent_word_count} in the text.
        Overall word count: {self.final_word_count}.
        Execution time: {self.get_time()} ms.
        '''
        return string

    def get_time(self) -> float:
        #value = round((self.end_time - self.start_time)/int(1e6), 3)
        return round((self.end_time - self.start_time)/int(1e6), 3)
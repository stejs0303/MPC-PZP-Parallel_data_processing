from math import inf

class Collected_data:
    type: str
    threads: int
    
    most_frequent_word: str
    most_frequent_word_count: int
    least_frequent_word: str
    least_frequent_word_count: int
    final_word_count: int
    
    preparation_time: int
    memory_manipulation_time: int
    processing_time: int
    execution_time: int
    
    def __init__(self, type:str = "", threads: int = 0,
                 most_frequent_word: str = "", most_frequent_word_count: int = 0, 
                 least_frequent_word: str = "", least_frequent_word_count: int = inf, 
                 final_word_count: int = 0, start_time: int = 0, prepared_time: int = 0, 
                 memory_coppied_allocated_time: int = 0, filtered_time: int = 0, stop_time: int = 0, 
                 compiling_start_time: int = 0, compiling_stop_time: int = 0):
        
        self.type = type
        self.threads = threads
        
        self.most_frequent_word = most_frequent_word
        self.most_frequent_word_count = most_frequent_word_count
        self.least_frequent_word = least_frequent_word
        self.least_frequent_word_count = least_frequent_word_count
        self.final_word_count = final_word_count
        
        self.preparation_time = round((prepared_time - start_time)*1000, 3)
        
        self.memory_manipulation_time = round((memory_coppied_allocated_time - prepared_time)*1000, 3)
        
        self.processing_time = round(((filtered_time - memory_coppied_allocated_time) - 
                                      (compiling_stop_time - compiling_start_time))*1000, 3)
        
        self.execution_time = round(((stop_time - start_time) - 
                                     (compiling_stop_time - compiling_start_time))*1000, 3)

    def __iadd__(self, other):
        self.type = other.type
        self.threads = other.threads
        
        self.most_frequent_word = other.most_frequent_word
        self.most_frequent_word_count = other.most_frequent_word_count
        self.least_frequent_word = other.least_frequent_word
        self.least_frequent_word_count = other.least_frequent_word_count
        self.final_word_count = other.final_word_count
        
        self.preparation_time = round((self.preparation_time + other.preparation_time) / 2, 2)
        self.memory_manipulation_time = round((self.memory_manipulation_time + other.memory_manipulation_time) / 2, 2)
        self.processing_time = round((self.processing_time + other.processing_time) / 2, 4)
        self.execution_time = round((self.execution_time + other.execution_time) / 2, 2)
        
        return self

    def __repr__(self) -> str:
        repr =  f'''
        Processed on: {self.type}, number of threads: {self.threads}.
        Most frequent word: \"{self.most_frequent_word}\", appeared: {self.most_frequent_word_count} in the text.
        Least frequent word: \"{self.least_frequent_word}\", appeared: {self.least_frequent_word_count} in the text.
        Overall word count: {self.final_word_count}.
        Measured times - Data preparation: {self.preparation_time} ms,
                         Memory manipulation: {self.memory_manipulation_time} ms,
                         Data processing: {self.processing_time} ms,
                         Execution time: {self.execution_time} ms.'''
        return repr
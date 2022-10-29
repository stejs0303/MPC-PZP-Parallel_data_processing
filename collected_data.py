from math import inf

class Collected_data:
    type: str
    threads: int
    
    most_frequent_word: str
    most_frequent_word_count: int
    least_frequent_word: str
    least_frequent_word_count: int
    final_word_count: int
    
    start_time: int
    prepared_time: int
    memory_coppied_allocated_time: int
    compiling_start_time: int
    compiling_stop_time: int
    filtered_time: int
    stop_time: int
    
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
        
        self.start_time = start_time
        self.prepared_time = prepared_time
        self.memory_coppied_allocated_time = memory_coppied_allocated_time
        self.filtered_time = filtered_time
        self.stop_time = stop_time
        self.compiling_start_time = compiling_start_time
        self.compiling_stop_time = compiling_stop_time

    def __repr__(self) -> str:
        return f'''
        Processed on: {self.type}, number of threads: {self.threads}.
        Most frequent word: \"{self.most_frequent_word}\", appeared: {self.most_frequent_word_count} in the text.
        Least frequent word: \"{self.least_frequent_word}\", appeared: {self.least_frequent_word_count} in the text.
        Overall word count: {self.final_word_count}.
        Measured times - Data preparation: {self.get_preparation_time()} ms,
                         Memory manipulation: {self.get_memory_manipulation_time()} ms,
                         Data processing: {self.get_processing_time()} ms,
                         Execution time: {self.get_execution_time()} ms.'''

    def get_preparation_time(self) -> float:
        return round((self.prepared_time - self.start_time)/int(1e6), 2)

    def get_memory_manipulation_time(self) -> float:
        return round((self.memory_coppied_allocated_time - self.prepared_time)/int(1e6), 2)
    
    def get_processing_time(self) -> float:
        return round((self.filtered_time - self.memory_coppied_allocated_time)/int(1e6) - self._get_compilation_time(), 4)
    
    def get_execution_time(self) -> float:
        return round((self.stop_time - self.start_time)/int(1e6) - self._get_compilation_time(), 2)
    
    def _get_compilation_time(self) -> float:
        return ((self.compiling_stop_time - self.compiling_start_time)/int(1e6))
    
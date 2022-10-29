from collected_data import Collected_data
from pycuda.compiler import SourceModule
import pycuda.driver as cuda_driver
from math import ceil
from load import load
import numpy as np
import time

MINIMUM_LETTERS: int = 4
MAXIMUM_LETTERS: int = 8

def multi_threaded(data: list, stop_words: list):
    cuda_driver.init()
    device = cuda_driver.Device(0)
    ctx = device.make_context()

    word_to_number: dict = {}
    number_to_word: dict = {}
    filtered_data: dict = {}
    encoded_words: list = []
    encoded_stop_words: list = []
    
    start = time.time_ns()
    
    for idx, stop_word in enumerate(stop_words):
        encoded_stop_words.append(idx)
        
        word_to_number[stop_word] = word_to_number.get(stop_word, int(len(word_to_number)))
    
    for word in data:
        if len(word) >= MINIMUM_LETTERS and len(word) <= MAXIMUM_LETTERS:
            word_to_number[word] = word_to_number.get(word, int(len(word_to_number)))
            number_to_word[word_to_number.get(word)] = word_to_number.get(word_to_number.get(word), word)
            encoded_words.append(word_to_number.get(word))
    
    #Add empty character to replace stop words with
    word_to_number[""] = word_to_number.get("", int(len(word_to_number)))
    number_to_word[word_to_number.get("")] = word_to_number.get(word_to_number.get(""), "")
    
    data_prepared = time.time_ns()
    
    encoded_stop_words: np.ndarray = np.array(encoded_stop_words).astype(np.int32)
    encoded_words: np.ndarray = np.array(encoded_words).astype(np.int32)
    filtered_data_numbers: np.ndarray = np.zeros(len(encoded_words)).astype(np.int32)
    filtered_data_info: np.ndarray = np.array([0, 0, 10000, 10000, 0]).astype(np.int32)
    # 0.most_frequent_word, 1.most_frequent_word_count, 2.least_frequent_word, 3.least_frequent_word_count, 4.counter
    
    data_gpu = cuda_driver.mem_alloc_like(encoded_words)
    stop_words_gpu = cuda_driver.mem_alloc_like(encoded_stop_words)
    filtered_data_gpu = cuda_driver.mem_alloc_like(filtered_data_numbers)
    filtered_data_info_gpu = cuda_driver.mem_alloc_like(filtered_data_info)

    cuda_driver.memcpy_htod(stop_words_gpu, encoded_stop_words)
    cuda_driver.memcpy_htod(data_gpu, encoded_words)
    cuda_driver.memcpy_htod(filtered_data_info_gpu, filtered_data_info)
    
    memory_coppied_allocated = time.time_ns()
    
    BLOCK_SIZE = 1024
    GRID_DIM = ceil(len(encoded_words)/BLOCK_SIZE)
    
    compiling_start = time.time_ns()
    
    kernel = SourceModule(get_kernel(BLOCK_SIZE, len(encoded_words), len(encoded_stop_words)))
    run = kernel.get_function("filter_data")
    
    compiling_stop = time.time_ns()
    
    try:
        run(data_gpu, stop_words_gpu, filtered_data_gpu, 
            filtered_data_info_gpu, np.int32(word_to_number.get("")), 
            block=(BLOCK_SIZE, 1, 1), grid=(GRID_DIM, 1, 1))
    except:
        stop_words_gpu.free()
        data_gpu.free()
        filtered_data_gpu.free()
        filtered_data_info_gpu.free()
        ctx.pop()
        exit(-1)
        
    ctx.synchronize()
    
    cuda_driver.memcpy_dtoh(filtered_data_numbers, filtered_data_gpu)
    cuda_driver.memcpy_dtoh(filtered_data_info, filtered_data_info_gpu)
    
    data_filtered = time.time_ns()
    
    for number in filtered_data_numbers:
        filtered_data[number_to_word[number]] = filtered_data.get(number_to_word[number], 0) + 1
    
    filtered_data.pop("")
    
    most_frequent_word = max(filtered_data, key=filtered_data.get)
    most_frequent_word_count = filtered_data.get(most_frequent_word)
    
    least_frequent_word = min(filtered_data, key=filtered_data.get)
    least_frequent_word_count = filtered_data.get(least_frequent_word)
    
    stop = time.time_ns()
    
    stop_words_gpu.free()
    data_gpu.free()
    filtered_data_gpu.free()
    filtered_data_info_gpu.free()
    
    ctx.pop()
    
    return Collected_data("GPU", f"{BLOCK_SIZE} per block", most_frequent_word, 
                          most_frequent_word_count, least_frequent_word, 
                          least_frequent_word_count, filtered_data_info[4], 
                          start, data_prepared, memory_coppied_allocated, data_filtered, stop,
                          compiling_start, compiling_stop)
                          
def get_kernel(BLOCK_SIZE: int, DATA_LENGTH: int, STOP_WORDS_LENGTH: int):
    
    kernel = '''
    __global__ void filter_data(int* in_data, int* in_stop_words, int* out_filtered_data, int* info, int empty_string)
    {   
        __shared__ int s_counter;
        __shared__ int s_stop_words[STOP_WORDS_LENGTH];
        __shared__ int s_data[BLOCK_SIZE];

        int idx = blockIdx.x * blockDim.x + threadIdx.x; 
     
        if(idx < DATA_LENGTH)
        {
            if(threadIdx.x == 0) s_counter = 0;
            if(threadIdx.x < STOP_WORDS_LENGTH) s_stop_words[threadIdx.x] = in_stop_words[threadIdx.x];
            s_data[threadIdx.x] = in_data[idx];
            
            __syncthreads();   
            
            int l_word = s_data[threadIdx.x];
            int l_initial = l_word;
            for(int i = 0; i < STOP_WORDS_LENGTH; i++)
                l_word = l_word + (l_word == s_stop_words[i]) * (empty_string - l_word);
            
            s_data[threadIdx.x] = l_word;
            
            atomicAdd(&s_counter, (l_initial == l_word));
            
            out_filtered_data[idx] = s_data[threadIdx.x];
            
            __syncthreads();          
            if(threadIdx.x == 0) atomicAdd(&info[4], s_counter);
        }    
    }
    '''
    
    kernel = kernel.replace("BLOCK_SIZE", str(BLOCK_SIZE))
    kernel = kernel.replace("DATA_LENGTH", str(DATA_LENGTH))
    kernel = kernel.replace("STOP_WORDS_LENGTH", str(STOP_WORDS_LENGTH))
    return kernel

if __name__=="__main__":
    data, stop_words = load("./files/data.txt", "./files/stop_words.txt")
    print(multi_threaded(data, stop_words))
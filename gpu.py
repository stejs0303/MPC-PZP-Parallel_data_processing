from collected_data import Collected_data
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import numpy as np
import time

MINIMUM_LETTERS: int = 4
MAXIMUM_LETTERS: int = 8

def multi_threaded(data: list, stop_words: list):
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()

    word_to_number: dict = {}
    number_to_word: dict = {}
    stop_words_numbers: list = []
    data_numbers: list = []
    
    start = time.time_ns()
    
    for idx, stop_word in enumerate(stop_words):
        stop_words_numbers.append(idx)
        
        word_to_number[stop_word] = word_to_number.get(stop_word, int(len(word_to_number)))
    
    for word in data:
        if len(word) >= MINIMUM_LETTERS and len(word) <= MAXIMUM_LETTERS:
            word_to_number[word] = word_to_number.get(word, int(len(word_to_number)))
            number_to_word[word_to_number.get(word)] = word_to_number.get(word_to_number.get(word), word)
            data_numbers.append(word_to_number.get(word))
    
    #Add empty character to replace stop words with
    word_to_number[""] = word_to_number.get("", int(len(word_to_number)))
    number_to_word[word_to_number.get("")] = word_to_number.get(word_to_number.get(""), "")
    
    stop_words_numbers: np.ndarray = np.array(stop_words_numbers).astype(np.int32)
    data_numbers: np.ndarray = np.array(data_numbers).astype(np.int32)
    filtered_data_numbers: np.ndarray = np.zeros(len(data_numbers)).astype(np.int32)
    filtered_data_info: np.ndarray = np.array([0, 0, 10000, 10000, 0]).astype(np.int32) #most_frequent_word, most_frequent_word_count, least_frequent_word, least_frequent_word_count, counter
    
    data_gpu = cuda.mem_alloc_like(data_numbers)
    stop_words_gpu = cuda.mem_alloc_like(stop_words_numbers)
    filtered_data_gpu = cuda.mem_alloc_like(filtered_data_numbers)
    filtered_data_info_gpu = cuda.mem_alloc_like(filtered_data_info)

    cuda.memcpy_htod(stop_words_gpu, stop_words_numbers)
    cuda.memcpy_htod(data_gpu, data_numbers)
    cuda.memcpy_htod(filtered_data_info_gpu, filtered_data_info)
    
    
    BLOCK_SIZE = 1024
    GRID_DIM = int(len(data_numbers)/BLOCK_SIZE)
    
    kernel = SourceModule(get_kernel(BLOCK_SIZE, len(data_numbers), len(stop_words_numbers)))
    run = kernel.get_function("kernel")
    
    run(data_gpu, stop_words_gpu, filtered_data_gpu, 
        filtered_data_info_gpu, np.int32(word_to_number.get("")), 
        block=(BLOCK_SIZE, 1, 1), grid=(GRID_DIM, 1, 1))
    
    ctx.syncronize()
    
    cuda.memcpy_dtoh(filtered_data_numbers, filtered_data_gpu)
    cuda.memcpy_dtoh(filtered_data_info, filtered_data_info_gpu)
    
    stop = time.time_ns()
    
    stop_words_gpu.free()
    data_gpu.free()
    filtered_data_gpu.free()
    filtered_data_info_gpu.free()
    ctx.pop()
    
    return Collected_data("GPU", start_time=start, end_time=stop)

def get_kernel(BLOCK_SIZE: int, DATA_LENGTH: int, STOP_WORDS_LENGTH: int):
    
    kernel = '''
    __global__ void kernel(int* data, int* stop_words, int* filtered_data, int* info, int replacer)
    {
        __shared__ int s_counter = 0;
        __shared__ int s_stop_words[];        
        __shared__ int s_data[BLOCK_SIZE];
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        int replace_with = replacer;
        
        if(idx < DATA_LENGTH)
        {
            s_stop_words = &stop_words; 
            s_data[threadIdx.x] = data[idx];
            
            __syncthreads();
            
            int l_data = s_data[idx];
            for(int i = 0; i < STOP_WORDS_LENGTH; i++)
                l_data = l_data + (l_data == s_stop_words[i]) * (replace_with - l_data);
            s_data[idx] = l_data;
            
        }
    }
    '''
    kernel = kernel.replace("BLOCK_SIZE", str(BLOCK_SIZE))
    kernel = kernel.replace("DATA_LENGTH", str(DATA_LENGTH))
    kernel = kernel.replace("STOP_WORDS_LENGTH", str(STOP_WORDS_LENGTH))
    return kernel
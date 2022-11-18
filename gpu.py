from collected_data import Collected_data
from pycuda.compiler import SourceModule
import pycuda.driver as cuda_driver
from math import ceil, inf
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
    encoded_words: list = []
    encoded_stop_words: list = []
    
    start = time.perf_counter()
    
    for idx, stop_word in enumerate(stop_words):
        encoded_stop_words.append(idx)
        word_to_number[stop_word] = word_to_number.get(stop_word, int(len(word_to_number)))
    
    for word in data:
        if len(word) >= MINIMUM_LETTERS and len(word) <= MAXIMUM_LETTERS:
            word_to_number[word] = word_to_number.get(word, int(len(word_to_number)))
            number_to_word[word_to_number.get(word)] = word_to_number.get(word_to_number.get(word), word)
            encoded_words.append(word_to_number.get(word))
            
    word_to_number[""] = word_to_number.get("", int(len(word_to_number)))
    number_to_word[word_to_number.get("")] = word_to_number.get(word_to_number.get(""), "")
    
    data_prepared = time.perf_counter()
    
    encoded_words: np.ndarray = np.array(encoded_words).astype(np.int32)
    encoded_stop_words: np.ndarray = np.array(encoded_stop_words).astype(np.int32)
    filtered_data_encoded: np.ndarray = np.zeros(len(encoded_words)).astype(np.int32)
    #data_info: np.ndarray = np.array([0, 0, 10000, 10000, 0]).astype(np.int32)
    histogram: np.ndarray = np.zeros(len(word_to_number)).astype(np.int32)
    
    data_gpu = cuda_driver.mem_alloc_like(encoded_words)
    stop_words_gpu = cuda_driver.mem_alloc_like(encoded_stop_words)
    filtered_data_gpu = cuda_driver.mem_alloc_like(filtered_data_encoded)
    #data_info_gpu = cuda_driver.mem_alloc_like(data_info)
    histogram_gpu = cuda_driver.mem_alloc_like(histogram)

    cuda_driver.memcpy_htod(stop_words_gpu, encoded_stop_words)
    cuda_driver.memcpy_htod(data_gpu, encoded_words)
    #cuda_driver.memcpy_htod(data_info_gpu, data_info)
    cuda_driver.memcpy_htod(histogram_gpu, histogram)
    
    memory_coppied_allocated = time.perf_counter()
    
    BLOCK_SIZE = 1024
    GRID_DIM = ceil(len(encoded_words)/BLOCK_SIZE)
    
    compiling_start = time.perf_counter()
    
    kernel = SourceModule(get_kernel(BLOCK_SIZE, len(encoded_words), len(encoded_stop_words), len(word_to_number)))

    compiling_stop = time.perf_counter()
    
    run = kernel.get_function("filter_data")  
    run(data_gpu, stop_words_gpu, filtered_data_gpu, 
        np.int32(word_to_number.get("")), 
        block=(BLOCK_SIZE, 1, 1), grid=(GRID_DIM, 1, 1))
        
    ctx.synchronize()
    
    run = kernel.get_function("create_histogram") 
    run(filtered_data_gpu, histogram_gpu, 
        np.int32(word_to_number.get("")), 
        block=(BLOCK_SIZE, 1, 1), grid=(GRID_DIM, 1, 1))
    
    ctx.synchronize()
    
    #run = kernel.get_function("get_info")
    #run(histogram_gpu, data_info_gpu,
    #    block=(BLOCK_SIZE, 1, 1), grid=(GRID_DIM, 1, 1))

    cuda_driver.memcpy_dtoh(histogram, histogram_gpu)

    data_filtered = time.perf_counter()
    
    most_frequent_word_count, most_frequent_word = 0, 0
    least_frequent_word_count, least_frequent_word = inf, 0
    counter = 0
    for word, num_of_occurences in enumerate(histogram):
        if most_frequent_word_count < num_of_occurences:
            most_frequent_word_count = num_of_occurences
            most_frequent_word = word
        elif least_frequent_word_count > num_of_occurences and num_of_occurences != 0:
            least_frequent_word_count = num_of_occurences
            least_frequent_word = word
        counter += num_of_occurences

    stop = time.perf_counter()
    
    stop_words_gpu.free()
    data_gpu.free()
    histogram_gpu.free()
    filtered_data_gpu.free()
    
    ctx.pop()
    
    return Collected_data("GPU", f"{BLOCK_SIZE} per block", 
                          number_to_word[most_frequent_word], most_frequent_word_count, 
                          number_to_word[least_frequent_word], least_frequent_word_count, counter, 
                          start, data_prepared, memory_coppied_allocated, data_filtered, stop,
                          compiling_start, compiling_stop)
                          
def get_kernel(BLOCK_SIZE: int, DATA_LENGTH: int, STOP_WORDS_LENGTH: int, UNIQUE_WORDS: int):
    
    kernel = '''
    __global__ void filter_data(int* in_data, int* in_stop_words, int* out_filtered_data, int empty_string)
    {   
        __shared__ int s_stop_words[STOP_WORDS_LENGTH];
        __shared__ int s_data[BLOCK_SIZE];

        int idx = blockIdx.x * blockDim.x + threadIdx.x; 
     
        if(idx < DATA_LENGTH)
        {
            if(threadIdx.x < STOP_WORDS_LENGTH) s_stop_words[threadIdx.x] = in_stop_words[threadIdx.x];
            s_data[threadIdx.x] = in_data[idx];
            
            __syncthreads();
            
            int l_word = s_data[threadIdx.x];
            for(int i = 0; i < STOP_WORDS_LENGTH; i++)
                l_word = l_word + (empty_string - l_word) * (l_word == s_stop_words[i]);
                 
            out_filtered_data[idx] = l_word;
        }    
    }
    
    __global__ void create_histogram(int* in_filtered_data, int* out_histogram, int empty_string)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;      
        
        __shared__ int s_filtered_data[BLOCK_SIZE];

        if(idx < DATA_LENGTH)
        {
            s_filtered_data[threadIdx.x] = in_filtered_data[idx];
            //__syncthreads();
            
            int l_word = s_filtered_data[threadIdx.x];
            atomicAdd(&out_histogram[l_word], (l_word != empty_string));
        }
    }
    
    __global__ void get_info(int* in_histogram, int* info)
    {
        __shared__ int s_histogram[BLOCK_SIZE];
        __shared__ int s_counter;
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if(idx < UNIQUE_WORDS)
        {
            s_histogram[threadIdx.x] = in_histogram[idx];
            if(threadIdx.x == 0) s_counter = 0;
            
            __syncthreads();
            
            atomicMax(&info[1], s_histogram[threadIdx.x]);
            atomicMin(&info[3], s_histogram[threadIdx.x] + (s_histogram[threadIdx.x] == 0) * 1000);
            atomicAdd(&s_counter, s_histogram[threadIdx.x]);
    
            __syncthreads();
        
            if(idx == 0)
            {
                info[0] = info[1];
                info[2] = info[3];
            }
            __syncthreads();
            
            atomicCAS(&info[0], s_histogram[threadIdx.x], idx);
            atomicCAS(&info[2], s_histogram[threadIdx.x], idx);
            
            if(threadIdx.x == 0) atomicAdd(&info[4], s_counter);
        }      
    }
    '''
    
    kernel = kernel.replace("BLOCK_SIZE", str(BLOCK_SIZE))
    kernel = kernel.replace("DATA_LENGTH", str(DATA_LENGTH))
    kernel = kernel.replace("STOP_WORDS_LENGTH", str(STOP_WORDS_LENGTH))
    kernel = kernel.replace("UNIQUE_WORDS", str(UNIQUE_WORDS))
    
    return kernel

if __name__=="__main__":
    data, stop_words = load("./files/data.txt", "./files/stop_words.txt")
    print(multi_threaded(data, stop_words))
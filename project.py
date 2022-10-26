from collected_data import Collected_data
import cpu_singlethreaded
import cpu_multithreaded
import gpu_multithreaded
import load as ld

single_thread: Collected_data
multi_thread: Collected_data
gpu: Collected_data

def main():
    data, stop_words = ld.load("./files/data.txt", "./files/stop_words.txt")
    
    single_thread = cpu_singlethreaded.single_thread(data, stop_words)
    
    print(single_thread)
    
if __name__ == "__main__":
    main()
    

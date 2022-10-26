#Author: Jan Stejskal
#ID: 211272
#Python version: 3.10.7
#Platform: Windows 11
#Packages: requirements.txt

from collected_data import Collected_data
import cpu
import gpu
import load as ld

def main():
    data, stop_words = ld.load("./files/data.txt", "./files/stop_words.txt")
    
    single_thread = cpu.single_threaded(data, stop_words)
    multi_thread = cpu.multi_threaded(data, stop_words)
    
    print(single_thread)
    
if __name__ == "__main__":
    main()
    

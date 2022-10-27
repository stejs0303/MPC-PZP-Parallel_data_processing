#Author:            Jan Stejskal
#School ID:         211272
#Python version:    3.10.7
#Platform:          Windows 11
#Packages:          requirements.txt

import load as ld
import cpu
import gpu

def main():
    data, stop_words = ld.load("./files/data.txt", "./files/stop_words.txt")
    
    single_thread = cpu.single_threaded(data, stop_words)
    multi_thread = cpu.multi_threaded(data, stop_words)
    
    print(single_thread)
    print(multi_thread)
    
if __name__ == "__main__":
    main()
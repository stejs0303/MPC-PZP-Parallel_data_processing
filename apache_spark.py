from collected_data import Collected_data
from pyspark import SparkContext
from operator import add
from load import load
from math import inf
import findspark
import time

MINIMUM_LETTERS: int = 4
MAXIMUM_LETTERS: int = 8 
    
def spark(data: list, stop_words: list):
    findspark.init()
    sc = SparkContext("local[12]")
    sc.setLogLevel("ERROR")
    
    start = time.perf_counter()
    
    spark_data = sc.parallelize(data)
    spark_filtered_data = spark_data.filter(lambda word: 
                                            len(word) >= MINIMUM_LETTERS and 
                                            len(word) <= MAXIMUM_LETTERS and 
                                            word not in stop_words)
    
    spark_pairs = spark_filtered_data.map(lambda word: (word, 1))
    spark_dict = spark_pairs.reduceByKey(add)
    filtered_data = spark_dict.collectAsMap()
    
    data_filtered = time.perf_counter()
       
    most_frequent_word = max(filtered_data, key=filtered_data.get)
    most_frequent_word_count = filtered_data.get(most_frequent_word)
    
    least_frequent_word = min(filtered_data, key=filtered_data.get)
    least_frequent_word_count = filtered_data.get(least_frequent_word) 
    counter = spark_filtered_data.count()
    
    stop = time.perf_counter()
    
    return Collected_data("Apache Spark", 12, most_frequent_word, most_frequent_word_count, 
                          least_frequent_word, least_frequent_word_count, counter, 
                          start, start, start, data_filtered, stop)

if __name__ == "__main__":
    data, stop_words = load("./files/data.txt", "./files/stop_words.txt")
    info = spark(data, stop_words)
    print(info)
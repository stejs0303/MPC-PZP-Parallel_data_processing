from pyspark import SparkContext, SparkConf
from collected_data import Collected_data
from operator import add
from load import load
import findspark
import time
    
def spark(data: list, stop_words: list):
    findspark.init()
    
    params = [("spark.driver.memory", "4g"), 
              ("spark.serializer", "org.apache.spark.serializer.KryoSerializer"),
              ("spark.shuffle.manager", "SORT"),
              ("spark.shuffle.consolidateFiles", "true"),
              ("spark.shuffle.spill", "true"),
              ("spark.shuffle.memoryFraction", "0.75"),
              ("spark.storage.memoryFraction", "0.45"),
              ("spark.shuffle.spill.compress", "false"),
              ("spark.shuffle.compress", "false")]
    
    sc = SparkContext(conf=SparkConf().setMaster("local[*]").setAll(params))
    sc.setLogLevel("ERROR")
    
    start = time.perf_counter()
    
    spark_data = sc.parallelize(data)
    
    time_memory_manip = time.perf_counter()
    
    spark_filtered_data = spark_data.filter(lambda word: len(word) >= 4 and len(word) <= 8 and word not in stop_words)
    spark_pairs = spark_filtered_data.map(lambda word: (word, 1))
    spark_dict = spark_pairs.reduceByKey(add)
    
    time_data_filtered = time.perf_counter()
    
    most_frequent_word, most_frequent_word_count = spark_dict.takeOrdered(1, key=lambda x: -x[1])[0]
    least_frequent_word, least_frequent_word_count = spark_dict.takeOrdered(1)[0]

    counter = spark_filtered_data.count()
    
    stop = time.perf_counter()
    
    sc.stop()
    return Collected_data("Apache Spark", 12, most_frequent_word, most_frequent_word_count, 
                          least_frequent_word, least_frequent_word_count, counter, 
                          start, start, time_memory_manip, time_data_filtered, stop)

if __name__ == "__main__":
    data, stop_words = load("./files/data.txt", "./files/stop_words.txt")
    info = spark(data, stop_words)
    print(info)
    
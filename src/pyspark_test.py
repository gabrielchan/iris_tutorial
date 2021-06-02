# Purpose of this file is test that PySpark is working using a simple word count example
# Sample data was taken from a Udemy Course

from pyspark.sql import SparkSession
from pyspark.sql import functions

file_path = "/Users/gchb/Documents/Code/random/random-code/scikitlearn_with_pyspark/data/Book"
spark = SparkSession.builder.appName("WordCountSQL").getOrCreate()

book = spark.read.text(file_path)

words = book.select(functions.explode(functions.split(book.value, "\\W+")).alias("word"))
words.filter(words.word != "")
lowercaseWords = words.select(functions.lower(words.word).alias("word"))
wordCount = lowercaseWords.groupBy("word").count().sort("count")

wordCount.show()

spark.stop()


from pyspark.sql.functions import regexp_replace, lower, when
import re
import emoji
import string
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.sql.functions import udf, col



# Clean emojis from text
def strip_emoji(text):
    return emoji.demojize(text)

# Remove punctuations, links, mentions, and \r\n new line characters
def strip_all_entities(text):
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower()
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    banned_list = string.punctuation + 'Ã' + '±' + 'ã' + '¼' + 'â' + '»' + '§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text

# Clean hashtags at the end of the sentence and remove the # symbol from words in the middle of the sentence
def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet))
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet))
    return new_tweet2

# Filter special characters such as & and $ present in some words
def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) or ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

# Remove multiple spaces
def remove_mult_spaces(text):
    return re.sub("\s\s+", " ", text)


# Define UDFs for cleaning text
strip_emoji_udf = udf(strip_emoji, StringType())
strip_all_entities_udf = udf(strip_all_entities, StringType())
clean_hashtags_udf = udf(clean_hashtags, StringType())
filter_chars_udf = udf(filter_chars, StringType())
remove_mult_spaces_udf = udf(remove_mult_spaces, StringType())

def clean_data(df):
    # Only include records with reviews
    df = df.filter(df['reviewText'].isNotNull())

    # Handle Null values for votes and cast it to integer
    df = df.na.fill({"vote": "0"})
    df = df.withColumn("vote", df["vote"].cast(IntegerType()))

    df = df.withColumn("reviewText", strip_emoji_udf("reviewText"))
    df = df.withColumn("reviewText", strip_all_entities_udf("reviewText"))
    df = df.withColumn("reviewText", clean_hashtags_udf("reviewText"))
    df = df.withColumn("reviewText", filter_chars_udf("reviewText"))
    df = df.withColumn("reviewText", remove_mult_spaces_udf("reviewText"))

    df = df.withColumn("reviewText", regexp_replace("reviewText", "<br />", " "))
    df = df.withColumn("reviewText", regexp_replace("reviewText", ";", "."))
    # Replace '\n' with a space
    df = df.withColumn("reviewText", regexp_replace("reviewText", "\\n", " "))

    # Remove square brackets and any text within them
    df = df.withColumn("reviewText", regexp_replace("reviewText", "\[?\[.+?\]?\]", " "))

    # Remove sequences of three or more slashes
    df = df.withColumn("reviewText", regexp_replace("reviewText", "/{3,}", " "))

    # Remove remaining HTML character entities like &#...;
    df = df.withColumn("reviewText", regexp_replace("reviewText", "\&\#.+\&\#\d+?;", " "))
    df = df.withColumn("reviewText", regexp_replace("reviewText", "\d+\&\#\d+?;", " "))
    df = df.withColumn("reviewText", regexp_replace("reviewText", "\&\#\d+?;", " "))

    # Remove facial expressions
    df = df.withColumn("reviewText", regexp_replace("reviewText", "\:\|", ""))
    df = df.withColumn("reviewText", regexp_replace("reviewText", "\:\)", ""))
    df = df.withColumn("reviewText", regexp_replace("reviewText", "\:\(", ""))
    df = df.withColumn("reviewText", regexp_replace("reviewText", "\:\/", ""))

    # Replace multiple spaces with single space
    df = df.withColumn("reviewText", remove_mult_spaces_udf("reviewText"))

    # Convert 'reviewText' to lowercase
    df = df.withColumn("reviewText", lower(df["reviewText"]))

    positive_threshold = 4
    negative_threshold = 2

    # Convert overall rating to Positive, Neutral, or Negative
    df = df.withColumn("label",
                   when(df["overall"] >= positive_threshold, 2.0)
                   .when(df["overall"] > negative_threshold, 1.0)
                   .otherwise(0.0))

    return df




spark = SparkSession.builder.appName("DataPreprocessing").getOrCreate()

dataset_fp = "./Dataset/Electronics.json"
meta_data_fp = "./Dataset/meta_Electronics.json"

# Defining the schema for JSON data
json_schema_data = StructType([
    StructField("overall", FloatType(), True),
    StructField("vote", StringType(), True),
    StructField("reviewTime", StringType(), True),
    StructField("reviewerID", StringType(), True),
    StructField("asin", StringType(), True),
    StructField("reviewerName", StringType(), True),
    StructField("reviewText", StringType(), True),
    StructField("summary", StringType(), True),
])

df = spark.read.schema(json_schema_data).json(dataset_fp)

# exclude reviews and products about books
json_schema_meta_data = StructType([

    StructField("asin", StringType(), True),
     StructField("main_cat", StringType(), True)
])

meta_df = spark.read.schema(json_schema_meta_data).json(meta_data_fp)

# Filter asin values for products with main_cat 'Books' in meta data
books_asin = meta_df.filter(col('main_cat') == 'Books').select('asin')

# Drop records in df where asin is in books_asin
df = df.join(books_asin, 'asin', 'left_anti')


# Apply UDFs to DataFrame columns
df = clean_data(df)

# Uncomment below code to write cleaned data to json file
# df.write.json('./Dataset/Electronics_preprocessed.json')

# Uncomment below code to generate subset of data in train val test format for bert model training
# # Split the data into train, validation, and test sets

# positive_df = df.filter(col('label') == 2.0).limit(20000)
#
# neutral_df = df.filter(col('label') == 1.0).limit(20000)
#
# negative_df = df.filter(col('label') == 0.0).limit(20000)
#
# train_positive_df, val_positive_df, test_positive_df = positive_df.randomSplit([0.6, 0.2, 0.2], seed=42)
# train_neutral_df, val_neutral_df, test_neutral_df = neutral_df.randomSplit([0.6, 0.2, 0.2], seed=42)
# train_negative_df, val_negative_df, test_negative_df = negative_df.randomSplit([0.6, 0.2, 0.2], seed=42)
#
# # Concatenate the DataFrames
# train_df_spark = train_positive_df.union(train_neutral_df).union(train_negative_df)
# val_df_spark = val_positive_df.union(val_neutral_df).union(val_negative_df)
# test_df_spark = test_positive_df.union(test_neutral_df).union(test_negative_df)
#
# # Shuffle the DataFrame pycharm
# train_df_spark = train_df_spark.withColumn("rand", rand())
# train_df_spark = train_df_spark.orderBy("rand").drop("rand")
#
# val_df_spark = val_df_spark.withColumn("rand", rand())
# val_df_spark = val_df_spark.orderBy("rand").drop("rand")
#
# test_df_spark = test_df_spark.withColumn("rand", rand())
# test_df_spark = test_df_spark.orderBy("rand").drop("rand")
#
# # Convert to Pandas DataFrame
# train_df = train_df_spark.toPandas()
# val_df = val_df_spark.toPandas()
# test_df = test_df_spark.toPandas()
#
# # save
# train_df.to_csv('./Dataset/train.csv', index=False)
# val_df.to_csv('./Dataset/val.csv', index=False)
# test_df.to_csv('./Dataset/test.csv', index=False)








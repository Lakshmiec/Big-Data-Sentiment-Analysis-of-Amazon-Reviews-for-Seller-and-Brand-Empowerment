
import pyspark.sql.types as pyspark_types
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, ArrayType,MapType
import spacy
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import contractions
from pyspark.sql.functions import udf, collect_list, explode, col

# list of all possible product pronouns
stopwords_to_exclude = prod_pronouns = ['it', 'this', 'that', 'they', 'these', 'those']
# list of all possible negation words
negation_words = ['no', 'not', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'never', 'hardly', 'scarcely']
# contraction words are taken care of so there not added to the list

# Initialize spacy 'en' model
def init_spacy():
    nlp_ = spacy.load('en_core_web_lg')
    # We want to replace product pronouns with the product name for aspect extraction
    for word in stopwords_to_exclude:
        nlp_.vocab[word].is_stop = False
    return nlp_

# Initialize NLTK SentimentIntensityAnalyzer
def init_nltk():
    print("\nLoading NLTK....")
    try :
        sid = SentimentIntensityAnalyzer()
    except LookupError:
        print("Please install SentimentAnalyzer using : nltk.download('vader_lexicon')")
    print("NLTK successfully loaded")
    return sid

# UDF to fix contractions
def fix_contractions(text):
    # Apply contractions fix logic here
    # This is just a placeholder, replace it with the actual contractions logic
    return contractions.fix(text)

# Register the UDF
contractions_udf = udf(fix_contractions, StringType())


# UDF to perform aspect extraction
def aspect_extraction(review_list):
    rule1_pairs = []
    rule2_pairs = []
    rule3_pairs = []
    rule4_pairs = []
    rule5_pairs = []
    rule6_pairs = []
    rule7_pairs = []

    nlp = broadcast_nlp.value
    sid = broadcast_sid.value


    for text in review_list:
        doc = nlp(text)

        for token in doc:
            A = "999999"
            M = "999999"

            if token.dep_ == "amod" and not token.is_stop:
                M = token.text
                A = token.head.text
                # check if aspect is a partial aspect of a noun phrase (string) one line code
                # A = next((noun_phrase for noun_phrase in noun_phrases if A in noun_phrase), A)


                # add adverbial modifier of adjective (e.g. 'most comfortable headphones')
                M_children = token.children
                for child_m in M_children:
                    if (child_m.dep_ == "advmod"):
                        M_hash = child_m.text
                        M = M_hash + " " + M
                        break

                # negation in adjective, the "no" keyword is a 'det' of the noun (e.g. no interesting characters)
                A_children = token.head.children
                for child_a in A_children:
                    if (child_a.dep_ == "det" and child_a.text in negation_words ):
                        neg_prefix = 'not'
                        M = neg_prefix + " " + M
                        break

            if (A != "999999" and M != "999999"):
                if A in prod_pronouns:
                    A = "product"
                    # A = next((noun_phrase for noun_phrase in noun_phrases if A in noun_phrase),
                    #          A)
                dict1 = {"noun": A, "adj": M, "rule": 1, "polarity": sid.polarity_scores(M)['compound']}
                rule1_pairs.append(dict1)

            # print("--- SPACY : Rule 1 Done ---")

            # # SECOND RULE OF DEPENDANCY PARSE -
            # # M - Sentiment modifier || A - Aspect
            # Direct Object - A is a child of something with relationship of nsubj, while
            # M is a child of the same something with relationship of dobj
            # Assumption - A verb will have only one NSUBJ and DOBJ
            children = token.children
            A = "999999"
            M = "999999"
            add_neg_pfx = False
            for child in children:
                if (child.dep_ == "nsubj" and not child.is_stop):
                    A = child.text
                    # A = next((noun_phrase for noun_phrase in noun_phrases if A in noun_phrase),
                    #          A)
                    # check_spelling(child.text)

                if ((child.dep_ == "dobj" and child.pos_ == "ADJ") and not child.is_stop):
                    M = child.text
                    # check_spelling(child.text)

                if (child.dep_ == "neg"):
                    neg_prefix = child.text
                    add_neg_pfx = True

            if (add_neg_pfx and M != "999999"):
                M = neg_prefix + " " + M

            if (A != "999999" and M != "999999"):
                if A in prod_pronouns:
                    A = "product"
                    # A = next((noun_phrase for noun_phrase in noun_phrases if A in noun_phrase),
                    #          A)
                dict2 = {"noun": A, "adj": M, "rule": 2, "polarity": sid.polarity_scores(M)['compound']}
                rule2_pairs.append(dict2)

            # print("--- SPACY : Rule 2 Done ---")

            ## THIRD RULE OF DEPENDANCY PARSE -
            ## M - Sentiment modifier || A - Aspect
            ## Adjectival Complement - A is a child of something with relationship of nsubj, while
            ## M is a child of the same something with relationship of acomp
            ## Assumption - A verb will have only one NSUBJ and DOBJ
            ## "The sound of the speakers would be better. The sound of the speakers could be better" - handled using AUX dependency

            children = token.children
            A = "999999"
            M = "999999"
            add_neg_pfx = False
            for child in children:
                if (child.dep_ == "nsubj" and not child.is_stop):
                    A = child.text
                    # A = next((noun_phrase for noun_phrase in noun_phrases if A in noun_phrase),
                    #          A)
                    # check_spelling(child.text)

                if (child.dep_ == "acomp" and not child.is_stop):
                    M = child.text

                # example - 'this could have been better' -> (this, not better)
                if (child.dep_ == "aux" and child.tag_ == "MD"):
                    neg_prefix = "not"
                    add_neg_pfx = True

                if (child.dep_ == "neg"):
                    neg_prefix = child.text
                    add_neg_pfx = True

            if (add_neg_pfx and M != "999999"):
                M = neg_prefix + " " + M
                # check_spelling(child.text)

            if (A != "999999" and M != "999999"):
                if A in prod_pronouns:
                    A = "product"
                    # A = next((noun_phrase for noun_phrase in noun_phrases if A in noun_phrase),
                    #          A)
                dict3 = {"noun": A, "adj": M, "rule": 3, "polarity": sid.polarity_scores(M)['compound']}
                rule3_pairs.append(dict3)
                # rule3_pairs.append((A, M, sid.polarity_scores(M)['compound'],3))
            # print("--- SPACY : Rule 3 Done ---")

            ## FOURTH RULE OF DEPENDANCY PARSE -
            ## M - Sentiment modifier || A - Aspect

            # Adverbial modifier to a passive verb - A is a child of something with relationship of nsubjpass, while
            # M is a child of the same something with relationship of advmod

            # Assumption - A verb will have only one NSUBJ and DOBJ

            children = token.children
            A = "999999"
            M = "999999"
            add_neg_pfx = False
            for child in children:
                if ((child.dep_ == "nsubjpass" or child.dep_ == "nsubj") and not child.is_stop):
                    A = child.text
                    # A = next((noun_phrase for noun_phrase in noun_phrases if A in noun_phrase),
                    #          A)
                    # check_spelling(child.text)

                if (child.dep_ == "advmod" and not child.is_stop):
                    M = child.text
                    M_children = child.children
                    for child_m in M_children:
                        if (child_m.dep_ == "advmod"):
                            M_hash = child_m.text
                            M = M_hash + " " + child.text
                            break
                    # check_spelling(child.text)

                if (child.dep_ == "neg"):
                    neg_prefix = child.text
                    add_neg_pfx = True

            if (add_neg_pfx and M != "999999"):
                M = neg_prefix + " " + M

            if (A != "999999" and M != "999999"):
                if A in prod_pronouns:
                    A = "product"
                    # A = next((noun_phrase for noun_phrase in noun_phrases if A in noun_phrase),
                    #          A)
                dict4 = {"noun": A, "adj": M, "rule": 4, "polarity": sid.polarity_scores(M)['compound']}
                rule4_pairs.append(dict4)
                # rule4_pairs.append((A, M,sid.polarity_scores(M)['compound'],4)) # )

            # print("--- SPACY : Rule 4 Done ---")

            ## FIFTH RULE OF DEPENDANCY PARSE -
            ## M - Sentiment modifier || A - Aspect

            # Complement of a copular verb - A is a child of M with relationship of nsubj, while
            # M has a child with relationship of cop

            # Assumption - A verb will have only one NSUBJ and DOBJ

            children = token.children
            A = "999999"
            buf_var = "999999"
            for child in children:
                if (child.dep_ == "nsubj" and not child.is_stop):
                    A = child.text
                    # A = next((noun_phrase for noun_phrase in noun_phrases if A in noun_phrase),
                    #          A)
                    # check_spelling(child.text)

                if (child.dep_ == "cop" and not child.is_stop):
                    buf_var = child.text
                    # check_spelling(child.text)

            if (A != "999999" and buf_var != "999999"):
                if A in prod_pronouns:
                    A = "product"
                    # A = next((noun_phrase for noun_phrase in noun_phrases if A in noun_phrase),
                    #          A)
                dict5 = {"noun": A, "adj": token.text, "rule": 5, "polarity": sid.polarity_scores(M)['compound']}
                rule5_pairs.append(dict5)
                # rule5_pairs.append((A, token.text,sid.polarity_scores(token.text)['compound'],5))

            # print("--- SPACY : Rule 5 Done ---")

            ## SIXTH RULE OF DEPENDANCY PARSE -
            ## M - Sentiment modifier || A - Aspect
            ## Example - "It ok", "ok" is INTJ (interjections like bravo, great etc)

            children = token.children
            A = "999999"
            M = "999999"
            if (token.pos_ == "INTJ" and not token.is_stop):
                for child in children:
                    if (child.dep_ == "nsubj" and not child.is_stop):
                        A = child.text
                        # A = next((noun_phrase for noun_phrase in noun_phrases if A in noun_phrase),
                        #          A)
                        M = token.text
                        # check_spelling(child.text)

            if (A != "999999" and M != "999999"):
                if A in prod_pronouns:
                    A = "product"
                    # A = next((noun_phrase for noun_phrase in noun_phrases if A in noun_phrase),
                    #          A)
                dict6 = {"noun": A, "adj": M, "rule": 6, "polarity": sid.polarity_scores(M)['compound']}
                rule6_pairs.append(dict6)

                # rule6_pairs.append((A, M,sid.polarity_scores(M)['compound'],6))

            # print("--- SPACY : Rule 6 Done ---")

            ## SEVENTH RULE OF DEPENDANCY PARSE -
            ## M - Sentiment modifier || A - Aspect
            ## ATTR - link between a verb like 'be/seem/appear' and its complement
            ## Example: 'this is garbage' -> (this, garbage)

            children = token.children
            A = "999999"
            M = "999999"
            add_neg_pfx = False
            for child in children:
                if (child.dep_ == "nsubj" and not child.is_stop):
                    A = child.text
                    # A = next((noun_phrase for noun_phrase in noun_phrases if A in noun_phrase),
                    #          A)
                    # check_spelling(child.text)

                if ((child.dep_ == "attr") and not child.is_stop):
                    M = child.text
                    # check_spelling(child.text)

                if (child.dep_ == "neg"):
                    neg_prefix = child.text
                    add_neg_pfx = True

            if (add_neg_pfx and M != "999999"):
                M = neg_prefix + " " + M

            if (A != "999999" and M != "999999"):
                if A in prod_pronouns:
                    A = "product"
                    # A = next((noun_phrase for noun_phrase in noun_phrases if A in noun_phrase),
                    #          A)
                dict7 = {"noun": A, "adj": M, "rule": 7, "polarity": sid.polarity_scores(M)['compound']}
                rule7_pairs.append(dict7)
                # rule7_pairs.append((A, M,sid.polarity_scores(M)['compound'],7))



    # print("--- SPACY : Rules Done ---")
    aspects_with_modifiers = rule1_pairs + rule2_pairs + rule3_pairs + rule4_pairs + rule5_pairs + rule6_pairs + rule7_pairs
    # count frequency of noun(aspect)
    aspect_freq = {}
    for aspect in aspects_with_modifiers:
        if aspect['noun'] in aspect_freq:
            aspect_freq[aspect['noun']] += 1
        else:
            aspect_freq[aspect['noun']] = 1

    # select top 10 aspects ( can be less than 10 if there are less than 10 aspects)
    top_aspects = sorted(aspect_freq, key=aspect_freq.get, reverse=True)[:5]


    top_aspects_with_modifiers = {}

    # return top aspect with modifiers in dictionary
    for aspect in top_aspects:
        top_aspects_with_modifiers[aspect] =[]
        for aspect_with_modifier in aspects_with_modifiers:
            if aspect_with_modifier['noun'] == aspect:
                top_aspects_with_modifiers[aspect].append((aspect_with_modifier['adj'], aspect_with_modifier['polarity']))

    # code return top_aspects_with_modifiers as dictionary
    return top_aspects_with_modifiers

# UDF to perform aggregation based on length limit of 1 million characters for spacy
def aggregate_reviews(reviews):
    total_length = 0
    aggregated_reviews = []
    nlp = broadcast_nlp.value
    result_list = []
    for review in reviews:
        doc = nlp(review)
        total_length += len(doc.text)

        if total_length <= 1000000:
            aggregated_reviews.append(review)
        else:
            result_list.append(" ".join(aggregated_reviews))
            total_length = len(doc.text)
            aggregated_reviews = [review]

    if aggregated_reviews:
        result_list.append(" ".join(aggregated_reviews))
    return result_list


# Create SparkSession
spark = SparkSession.builder.appName("AspectExtraction").config("spark.executor.memory", "20g") \
    .config("spark.driver.memory", "20g").getOrCreate()


# Path to dataset with records whose having min 5 votes.
dataset_fp = "./Dataset/filtered_reviews.json"

json_schema_data = pyspark_types.StructType([
    StructField("overall", FloatType(), True),
    StructField("vote", StringType(), True),
    StructField("reviewTime", StringType(), True),
    StructField("reviewerID", StringType(), True),
    StructField("asin", StringType(), True),
    StructField("reviewerName", StringType(), True),
    StructField("reviewText", StringType(), True),
    StructField("summary", StringType(), True),
    StructField("label", FloatType(), True),
])

# Initialize spacy 'en' model
nlp = init_spacy()
broadcast_nlp = spark.sparkContext.broadcast(nlp)

# Read the dataset
df = spark.read.schema(json_schema_data).json(dataset_fp)

# Apply UDFs to DataFrame columns
df = df.withColumn("reviewText", contractions_udf(df["reviewText"]))

# Define the UDF schema
aggregate_reviews_udf = spark.udf.register("aggregate_reviews", aggregate_reviews, ArrayType(StringType()))

# Aggregate reviews based on product ID and apply the UDF
grouped_df = (
    df.groupBy("asin")
    .agg(aggregate_reviews_udf(collect_list("reviewText")).alias("reviews_list"))
)

# Initialize NLTK SentimentIntensityAnalyzer
sid = init_nltk()
broadcast_sid = spark.sparkContext.broadcast(sid)

# Define the UDF schema
aspect_extraction_schema = MapType(StringType(), ArrayType(StructType([
    StructField("adj", StringType(), False),
    StructField("polarity", FloatType(), False),
])))


# Register the UDF
aspect_extraction_spark_udf = udf(aspect_extraction, aspect_extraction_schema)

 # Apply the UDF
grouped_df = grouped_df.withColumn("aspects", aspect_extraction_spark_udf(grouped_df.reviews_list))

product_aspects = grouped_df.select("asin", "aspects")



# explode aspects column to get each aspect in a row with its modifiers and polarity score in a row for each aspect of a product.
product_aspects = product_aspects.select("asin", explode("aspects").alias("aspect", "modifiers"))
exploded_df = product_aspects.select("asin", "aspect", "modifiers").selectExpr("asin", "aspect", "explode(modifiers) as modifier")
exploded_df_final = exploded_df.select("asin", "aspect",  col("modifier.adj").alias("modifier"), col("modifier.polarity").alias("polarity"))

exploded_df_final = exploded_df_final.dropna()

# Uncomment code to show the final dataframe
#exploded_df_final.show(5)

# Uncomment code write to single json file
#exploded_df_final.coalesce(1).write.json("/Users/dhanushlalitha/Documents/SEM2/BIG_DATA/final_data/product_aspects.json")

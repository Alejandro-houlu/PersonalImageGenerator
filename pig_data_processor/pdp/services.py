import itertools
from pig_data_processor import settings
import os
import pandas as pd
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.metrics.pairwise import cosine_similarity

user_data_file_path = os.path.join(settings.STATIC_DIR, 'user_data','user_data_10k_rows.csv')
user_item_ratings_data_file_path = os.path.join(settings.STATIC_DIR, 'user_ratings_data','user_item_ratings_ver2.csv')
item_data_file_path = os.path.join(settings.STATIC_DIR, 'item_data','full_item_data_ver2.csv')
recommendation_file_outpath = os.path.join(settings.STATIC_DIR, 'recommendations')
user_recommendation_file_path = os.path.join(settings.STATIC_DIR, 'recommendations','user_recommendations_CF_ALS.csv')
als_model_file_path = os.path.join(settings.STATIC_DIR, 'models','cf_als')


def getUserInfoById(userId):
    print('Start of getUserInfoById service method')
    user_item_ratings_data = pd.read_csv(user_item_ratings_data_file_path, dtype=str)
    user_row_list = user_item_ratings_data[user_item_ratings_data['user_id'] == userId]

    if user_row_list.empty:
        return None
    
    item_ratings_map = dict(zip(user_row_list['item_ids'], user_row_list['ratings']))
    
    userId_and_purchaseHistory = {
        'userId' : userId,
        'item_ids': item_ratings_map
        
        }
    print('End of getUserInfoById service method')
    return userId_and_purchaseHistory

def getUserRecommendationByUserId(userId):
    print('Start of getUserRecommendationByUserId service method')

    user_item_ratings_data = pd.read_csv(user_recommendation_file_path, dtype=str)
    user_row_list = user_item_ratings_data[user_item_ratings_data['user_id'] == userId]
    ensure_https(user_row_list)

    if user_row_list.empty:
        return None
    
    # item_info_map = dict(zip(user_row_list['item_id'], user_row_list['item_url']))
    item_info_map = dict(zip(user_row_list['item_id'], [list(x) for x in zip(user_row_list['item_desc'], user_row_list['item_url'])]))
    
    user_recs = {
        'userId' : userId,
        'item_info': item_info_map
        
        }
    print('End of getUserRecommendationByUserId service method')
    return user_recs

def getItemData():
    return pd.read_csv(item_data_file_path,dtype=str)

def ensure_https(df):
    df['item_url'] = df['item_url'].apply(lambda url: (
        'https:' + url if url.startswith('//') else
        'https://' + url if not url.startswith('http') else
        url
    ))
    return df


def generate_user_recommendation():
    print('Start of user recommendation generation ----------------->')
    
    # Get items first for later use
    itemdata = getItemData()
    # item_descriptions = itemdata['item_desc_en'].tolist()
    # item_ids = itemdata['item_id'].tolist()

    titlelookup = dict(zip(itemdata["item_id"],itemdata["item_desc_en"])) # create a lookup dictionary
    # Print the first 2 items from the dictionary
    for i, (key, value) in enumerate(titlelookup.items()):
        if i < 2:
            print(key, value)
        else:
            break

    titlelookup_url = dict(zip(itemdata["item_id"],itemdata["item_url"])) # create a lookup dictionary
    # Print the first 2 items from the dictionary
    for i, (key, value) in enumerate(titlelookup_url.items()):
        if i < 2:
            print(key, value)
        else:
            break

    #Start the spark session
    spark = SparkSession.builder.getOrCreate()

    print("Spark session started successfully.")

    ratings_spdf = spark.read.csv(user_item_ratings_data_file_path, header=True)
    newcolnames = ['userid','itemid','rating']
    for c,n in zip(ratings_spdf.columns,newcolnames):
        ratings_spdf=ratings_spdf.withColumnRenamed(c,n)
    # print(ratings_spdf)
    
    # Cast the ratings to float so that it can be processed
    ratings_spdf = ratings_spdf.withColumn("rating", ratings_spdf.rating.cast("Float"))

    # We only want score that are between 2 and 12 because outside of that we have too much and too little items
    # ratings_spdf = ratings_spdf.filter((ratings_spdf['rating'] > 2) & (ratings_spdf['rating'] < 13))
    ratings_spdf = ratings_spdf.filter((ratings_spdf['rating'] < 13))

    # Get distinct userids and itemids
    distinct_userids = ratings_spdf.select('userid').distinct().collect()
    distinct_itemids = ratings_spdf.select('itemid').distinct().collect()
    # print(distinct_userids)
    

    # Create dictionaries to map userids and itemids to indices
    user_dict = {row.userid: index for index, row in enumerate(distinct_userids)}
    item_dict = {row.itemid: index for index, row in enumerate(distinct_itemids)}

    # These are for later use when finding 
    item_dict2 = {index: row.itemid for index, row in enumerate(distinct_itemids)}
    user_dict2 = {index: row.userid for index, row in enumerate(distinct_userids)}

    # copy the integer indices into the ratings dataframe
    rdd2=ratings_spdf.rdd.map(lambda x: (user_dict[x[0]],item_dict[x[1]],float(x[2])))
    ratings_spdf = rdd2.toDF()

    # reinsert the column names
    ratings_spdf = rdd2.toDF(["userid", "itemid", "rating"])

    # split data into training and test sets (these are also spark dataframes)
    (training, test) = ratings_spdf.randomSplit([0.8, 0.2])
    print(type(training))
    print(type(test))
    print("trainset=",training.count(), "test set=", test.count())

    # Create df to use for precision and recall evaluation
    test_set = test.toPandas()
    train_set = training.toPandas()

    # Fit the ALS model 
    als = ALS(maxIter=15, rank=100, regParam=0.1, userCol="userid", itemCol="itemid", ratingCol="rating", coldStartStrategy="drop", implicitPrefs=False)
    model = als.fit(training)
    # saveModel(model, 'cf_als_model')
    model.save(als_model_file_path)

    evalutate_model_mae(model,test)

    # Generate user recommendation
    userRecs = model.recommendForAllUsers(10)

    # Set the k for precision@k
    k=5
    userRecs_df = userRecs.toPandas()
    precision, recall = precision_recall_at_k(userRecs_df, test_set, k)

    print('k =', k)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    #Retrieve the recommendations in readable format
    retrieve_user_recommendations(userRecs, user_dict2, item_dict2,titlelookup,titlelookup_url)


def evalutate_model_mae(model,test):
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")
    error = evaluator.evaluate(predictions)
    print("Mean Absolute error = ", error)

def precision_recall_at_k(recs_df, test_df, k):
    
    precisions = []
    recalls = []

    for user_id in test_df['userid'].unique():
        # Get actual items bought by the user in the test set (ground truth)
        actual_items = test_df[test_df['userid'] == user_id]['itemid'].values
        # actual_items = np.array(actual_items)
        # print(actual_items)

        user_recs = recs_df[recs_df['userid'] == user_id]['recommendations']
        if not user_recs.empty:

            # Get top-k recommended items for the user
            recs = user_recs.values[0]
            recommended_items = [rec.itemid for rec in recs[:k]]
            recommended_items = np.array(recommended_items)

            actual_items_flat = actual_items.flatten().astype(int).tolist()
            recommended_items_flat = recommended_items.astype(int).tolist()

            # Calculate precision@k
            hits = len(set(recommended_items_flat).intersection(set(actual_items_flat)))
            precision = hits / len(recommended_items_flat) if len(recommended_items_flat) > 0 else 0

            # Calculate recall@k
            recall = hits / len(actual_items_flat) if len(actual_items_flat) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

    # Return average precision and recall
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)

    return avg_precision, avg_recall

def retrieve_user_recommendations(userRecs,user_dict,item_dict,itemLookup, itemlookup_url):
    userRecs2 = userRecs.collect()
    itemData = getItemData()
    records = []
    print('We are in retrieve user recs')
    for i in userRecs2:
        print(i)
        user_id = user_dict[i[0]]
        print("Recommend for User ID:", user_id)
        print()
        for rec in i[1]:
            item_id = item_dict[rec[0]]
            print(item_id)
            if item_id not in itemLookup:
                continue
            item_desc = itemLookup[item_id]
            item_url = itemlookup_url[item_id]
            print(item_desc)
            records.append({'user_id': user_id, 'item_id': item_id, 'item_desc':item_desc, 'item_url':item_url})
        print()
    
    df = pd.DataFrame(records)
    df.to_csv(recommendation_file_outpath + '/user_recommendations_CF_ALS.csv', index=False)
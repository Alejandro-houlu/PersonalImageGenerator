from datetime import datetime
from pig_data_processor import settings
import os
import pandas as pd
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import numpy as np
from tensorflow import keras
from openai import OpenAI
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer

from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from keras_preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.optimizers import Adam
import google.generativeai as genai
from django.shortcuts import render
from googleapiclient.discovery import build
import re


user_data_file_path = os.path.join(settings.STATIC_DIR, 'user_data','user_data_10k_rows.csv')
user_item_ratings_data_file_path = os.path.join(settings.STATIC_DIR, 'user_ratings_data','user_item_ratings_ver2.csv')
item_data_file_path = os.path.join(settings.STATIC_DIR, 'item_data','full_item_data_ver2.csv')
recommendation_file_outpath = os.path.join(settings.STATIC_DIR, 'recommendations')
user_recommendation_file_path = os.path.join(settings.STATIC_DIR, 'recommendations','user_recommendations_CF_ALS.csv')
als_model_file_path = os.path.join(settings.STATIC_DIR, 'models','cf_als')
lstm_embeddings_file_path = os.path.join(settings.STATIC_DIR, 'models','lstm_embeddings.csv')
user_gen_image_data_file_path = os.path.join(settings.STATIC_DIR, 'user_gen_image_data','user_gen_image_data.csv')


def getUserInfoById(userId):
    print('Start of getUserInfoById service method')
    user_item_ratings_data = pd.read_csv(user_item_ratings_data_file_path, dtype=str)
    itemData = getItemData()
    user_row_list = user_item_ratings_data[user_item_ratings_data['user_id'] == userId]
    
    if user_row_list.empty:
        return None
    merged_data = user_row_list.merge(itemData[['item_id', 'item_url','item_desc_en']], left_on='item_ids', right_on='item_id', how='inner')
    result = merged_data[['item_id','item_desc_en', 'ratings', 'item_url']].values.tolist()
    
    userId_and_purchaseHistory = {
        'result': result
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
    
    item_info_list = [
    [row['item_id'], row['item_desc'], row['item_url']] 
    for _, row in user_row_list.iterrows()
    ]
    user_recs = {
        'userId' : userId,
        'item_info': item_info_list
        
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
    spark.stop()
    return None


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

def retrieve_user_recommendations(userRecs,user_dict,item_dict,itemLookup,itemlookup_url):
    userRecs2 = userRecs.collect()
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
    return None

def train_lstm_model():
    print('Start of train_lstm_model services method')

    itemdata = getItemData()
    item_descriptions = itemdata['item_desc_en'].tolist()
    item_ids = itemdata['item_id'].tolist()

    # One-Hot Encoding of category_id
    encoder = OneHotEncoder(sparse=False)  # Use sparse=False for a dense array output
    category_encoded = encoder.fit_transform(itemdata[['cat_id']])

    # Check the number of unique categories
    num_categories = category_encoded.shape[1]
    print(f"Number of categories: {num_categories}")
    print("Shape of Category Encoded:", category_encoded.shape)

    # Tokenization and padding
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(item_descriptions)
    sequences = tokenizer.texts_to_sequences(item_descriptions)
    padded_sequences = pad_sequences(sequences, maxlen=50)  # Adjusted maxlen
    print("First 5 Padded Sequences:")
    print(padded_sequences[:5])
    print("Shape of Padded Sequences:", padded_sequences.shape)

    # LSTM Model
    lstm_model = Sequential()
    lstm_model.add(Embedding(input_dim=10000, output_dim=128, input_length=50))
    lstm_model.add(LSTM(128))
    lstm_model.add(Dense(75, activation='relu'))

    # Compile and get embeddings
    # lstm_model.compile(optimizer='adam')
    lstm_model.compile(optimizer=Adam(learning_rate=0.000005), loss='categorical_crossentropy', metrics=['accuracy'])
    # Fit the model (consider splitting into train/test sets for validation)

    # Check the model summary
    lstm_model.summary()  # Debugging output to check the model architecture

    lstm_model.fit(padded_sequences, category_encoded,epochs=10, batch_size=32)
    item_embeddings = lstm_model.predict(padded_sequences)

    # Create a DataFrame for LSTM embeddings
    lstm_embeddings_df = pd.DataFrame(item_embeddings)
    lstm_embeddings_df['item_id'] = item_ids

    lstm_embeddings_df.to_csv(lstm_embeddings_file_path, index=False)
    
    return None

def recommend_similar_items(item_id, num_similar_items,userId,item):
    #Start the spark session
    spark = SparkSession.builder.getOrCreate()
    print("Spark session started successfully.")

    itemdata = getItemData()
    titlelookup = dict(zip(itemdata["item_id"],itemdata["item_desc_en"])) # create a lookup dictionary
    titlelookup_url = dict(zip(itemdata["item_id"],itemdata["item_url"])) # create a lookup dictionary
    records = []

    searchedItem = itemdata[itemdata['item_id']==item_id]
    records.append({'Searched item Id': searchedItem['item_id'].values[0], 'item_id': searchedItem['item_id'].values[0], 'item_desc':searchedItem['item_desc_en'].values[0], 'item_url':searchedItem['item_url'].values[0]})


    # Load the combined ALS and LSTM embeddings data
    als_model = ALSModel.load(als_model_file_path)
    als_item_factors = als_model.itemFactors.toPandas()
    als_item_factors['item_id'] = als_item_factors['id'].apply(lambda x: itemdata['item_id'].iloc[x])
    lstm_embeddings_df = pd.read_csv(lstm_embeddings_file_path)  # Load saved LSTM embeddings
    
    # Merge on 'item_id' to get a combined DataFrame
    combined_df_lstm_als = pd.merge(als_item_factors, lstm_embeddings_df, on='item_id', how='inner')

    # Get the combined embedding for the target item
    target_row = combined_df_lstm_als[combined_df_lstm_als['item_id'] == item_id]  # Get the embedding for the given item
    if target_row.empty:
        print(f"Item ID {item_id} not found.")
        return []

    target_features_als = target_row['features'].values[0]
    target_embeddings_lstm = target_row.iloc[:, 3:78].values.flatten()
    
    # Normalize both feature vectors
    target_features_als = normalize_vector(target_features_als)
    target_embeddings_lstm = normalize_vector(target_embeddings_lstm)

    # Combine ALS and LSTM embeddings for the target item
    combined_features = np.concatenate([target_features_als, target_embeddings_lstm])

    # Prepare combined embeddings for all other items
    other_items_combined = []
    for _, row in combined_df_lstm_als.iterrows():
        features = row['features']  # ALS features
        embeddings = row.iloc[3:78].values.flatten()  # LSTM embeddings (columns 3 to 66)

        if isinstance(features, list):  # Check if features is a list, convert if needed
            features = np.array(features)

        combined = np.concatenate([features, embeddings])
        other_items_combined.append(combined)

    # Convert to a NumPy array for cosine similarity calculations
    other_items_combined = np.array(other_items_combined)

    # Calculate cosine similarity between target item and all items
    similarities = cosine_similarity([combined_features], other_items_combined)
    similar_items = np.argsort(-similarities)[:, :num_similar_items]  # Get top N most similar items
    similar_items_ids = combined_df_lstm_als.iloc[similar_items[0]]['item_id'].values
    print(similar_items_ids)
 
    for id in similar_items_ids:
        item_desc = titlelookup[id]
        item_url = titlelookup_url[id]
        records.append({'Searched item Id': item_id, 'item_id': id, 'item_desc':item_desc, 'item_url':item_url})

    searched_item_desc = titlelookup[item_id]
    search_item_url = titlelookup_url[item_id]
    print('Search item', item_id)
    print(searched_item_desc)
    print(search_item_url)
    top_item_recs = [records[1]['item_desc'], records[2]['item_desc']]
    print(userId)

    if item is None or str(item).strip()== "''":
        print('--------------------------------------------------------------------->')
        print(item)
        return records
    else:
        try:
            generate_prompt_using_recs(top_item_recs,userId,item)
        except Exception as ex:
            print(f"Error getting generating prompt using item recs: {str(ex)}")
        finally:
            return records

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm

def generate_prompt_using_recs(item_recs,userId,item):
    print('Start of generate prompt using recs ----------------->')
    print('Top items parsed:', item_recs)

    # Load the CSV file
    file_path = user_recommendation_file_path
    df = pd.read_csv(file_path)

    # Find the first occurrence of userId = 1
    first_user_1 = df[df['user_id'] == userId].iloc[0]
    item_recs.append(first_user_1['item_desc'])

    item_recs_string = ', '.join(item_recs)

    print('Items to be used for prompting ---------->', item_recs_string)
    print('Item searched by the user --------------->', item)

    # Set your API key
    gcp_api_key=os.getenv('GCP_KEY')
    print(gcp_api_key)
    genai.configure(api_key=gcp_api_key)
    # for model in genai.list_models():
    #     pprint.pprint(model)
    gemini = genai.GenerativeModel('gemini-1.5-flash')
    # Prompt for image generation
    prompt = item_recs_string + \
         ", given the 3 items above, study the distinctive features and in less than 20 words, use those features and make a '{}' ,make it look realistic and only one item",format(item)


    #Generate the image
    result = gemini.generate_content(prompt)

    print(result.text)
    print('End of generate prompt using recs ----------------->')

    try:
        generate_image(userId, result.text)
    except Exception as ex:
        print(f"Error generating image: {str(ex)}")
  

def generate_image(userId, prompt):

    openai_api_key=os.getenv('OPENAI_KEY')

    # Set your API key
    client = OpenAI(api_key=openai_api_key)

    # Generate an image
    response = client.images.generate(
        model= "dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024"
    )

    image_data = response.data
    image_url = image_data[0].url
    revised_prompt = image_data[0].revised_prompt
    print(image_url)
    print(response)

    log_gen_image_to_csv(userId, image_url,prompt,revised_prompt)

    gen_image_obj = {
        "user_id" : userId,
        "image_url" : image_url
    }

    return gen_image_obj

def log_gen_image_to_csv(user_id, image_url,prompt,revised_prompt):
    # Create a DataFrame with the new log entry
    new_entry = {
        'user_id': user_id,
        'gen_image_url': image_url,
        'prompt': clean_string(prompt),
        'revised_prompt': revised_prompt,
        'timestamp': datetime.now()
    }
    df = pd.DataFrame([new_entry])

    try:
        # Append the new entry to the existing CSV
        df.to_csv(user_gen_image_data_file_path, mode='a', header=not pd.io.common.file_exists(user_gen_image_data_file_path), index=False)
    except Exception as e:
        print("Error writing to CSV:", e)

def search_items(search_term, top_n=40):
    # Preprocessing: Convert search term to lowercase
    item_data = getItemData()
    search_term = search_term.lower()

    # Fuzzy match to get the closest item description
    item_descriptions = item_data['item_desc_en'].str.lower().tolist()
    closest_match = process.extractOne(search_term, item_descriptions)

    # If a close match is found, use that as the corrected term
    if closest_match and closest_match[1] > 60:  # You can adjust the threshold
        search_term = closest_match[0]

    # Add search term to the item data for comparison
    item_data['combined'] = item_data['item_desc_en'].str.lower()

    # Vectorize descriptions using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(item_data['combined'].tolist() + [search_term])

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Get the top N indices based on similarity scores
    top_indices = cosine_sim.argsort()[0][::-1][:top_n]

    result = {
        "result" : item_data.iloc[top_indices].values.tolist()
    }

    # Return the top items based on indices
    return result

def getGeneratedImages(userId):
    genImageData = pd.read_csv(user_gen_image_data_file_path,dtype=str)
    user_rows = genImageData[genImageData['user_id']==userId]
    result = user_rows.to_dict(orient='records')
    return result

def clean_string(input_string):
    # Replace commas with spaces
    cleaned_string = input_string.replace(',', ' ')
    
    # Remove all symbols (keeping only alphanumeric characters and spaces)
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_string)
    
    # Optionally, strip leading/trailing whitespace
    cleaned_string = cleaned_string.strip()
    
    return cleaned_string
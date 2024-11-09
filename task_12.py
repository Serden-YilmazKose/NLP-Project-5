from concurrent.futures import ProcessPoolExecutor, as_completed
from spellchecker import SpellChecker
from nltk.metrics import edit_distance
import os
from tqdm import tqdm
import math
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import json
from nltk.tokenize import RegexpTokenizer
import numpy as np
import pickle
tokenizer = RegexpTokenizer(r'\w+')

filename_corrections = 'corrections.csv'
filename_distances = 'distances.csv'

# Create stopwords list, and add words that do not belong in the vocabulary 
Stopwords = set(upper_word.lower() for upper_word in nltk.corpus.stopwords.words('english'))
Stopwords.add("http")
Stopwords.add("https")
Stopwords.add("com")
Stopwords.add("www")
Stopwords.add("r")
Stopwords.add("html")


def load_data_from_json(json_path):
    # Read json data from file
    with open(json_path, 'r') as f:
        json_data = f.readlines()
        
    # Parse json data
    json_data = [json.loads(line) for line in json_data]

    # Convert json data to pandas dataframe
    json_df = pd.DataFrame(json_data)
    
    return json_df

def get_tokens_by_category(data):
    # Extracting the bodies sorted by category into a dataframe
    bodies_by_category = {}

    for index, row in data.iterrows():
        # Create body and label integer and string
        body, label = str(row['body']), str(row['gold_label'])
        if label in bodies_by_category:
            bodies_by_category[label].append(body)
        else:
            bodies_by_category[label] = [body]

    bodies_by_category = pd.DataFrame(bodies_by_category.items(), columns=['Category', 'Bodies'])

    # Preprocessing the bodies and storing them in a new dataframe
    tokens_by_category = {}

    for index, row in bodies_by_category.iterrows():
        bodies, label = row['Bodies'], row['Category']
        for body in bodies:
            tokens = word_tokenize(body.lower())
            tokens = [word for word in tokens if word.isalnum()]
            if label in tokens_by_category:
                tokens_by_category[label].extend(tokens)
            else:
                tokens_by_category[label] = []
                tokens_by_category[label] = tokens

    tokens_by_category = pd.DataFrame(tokens_by_category.items(), columns=['Category', 'Tokens'])
    
    return tokens_by_category

def remove_stopwords(tokens_by_category):
    # Removing the stopwords from the tokens since we dont need those for empath categorization
    tokens_by_category_stopwords_removed = {}

    for index, row in tokens_by_category.iterrows():
        category = str(row['Category'])
        tokens = row['Tokens']

        tokens_by_category_stopwords_removed[category] = [word for word in tokens if word.lower() not in Stopwords]
        
    return tokens_by_category_stopwords_removed

# Initialize the spell checker
spell = SpellChecker()

# Function to find the closest correction and calculate edit distance for a chunk of tokens
def find_corrections_and_distances(category_name, tokens_chunk, already_found):
    corrections = []
    distances = []
    local_found = {}
    
    tokens_not_seen_before = 0

    # Identify misspelled words in the chunk (process each token separately)
    misspelled = []
    for token in tokens_chunk:
        if not spell.unknown([token]):
            continue
        misspelled.append(token)
    
    for word in misspelled:
        correction = None
        # Check if the word has already been found
        if word in local_found:
            correction = local_found[word]
        elif word in already_found:
            correction = already_found[word]
        else:
            # Correct the misspelled word
            correction = spell.correction(word)
            tokens_not_seen_before += 1
            if correction:
                local_found[word] = correction

        if correction:
            corrections.append((word, correction))
            distances.append(edit_distance(word, correction))
            
    # Print the number of tokens that were and were not seen before
    # print(f"Category: {category_name}, Tokens seen before: {len(misspelled) - tokens_not_seen_before}, Tokens not seen before: {tokens_not_seen_before}")
            
    # Return the category name, corrections, distances, and local found dictionary
    return category_name, corrections, distances, local_found

# Helper function to split tokens into chunks
def chunk_tokens(tokens, chunk_size):
    for i in range(0, len(tokens), chunk_size):
        yield tokens[i:i + chunk_size]

if __name__ == "__main__":
    
    # Load data from json file
    json_path = "./kurrek.2020.slur-corpus.json"
    data = load_data_from_json(json_path)
    
    # Make sure that the body is a string (from task 4 in jupyter notebook)
    data['body'] = data['body'].apply(lambda x: str(x) if not isinstance(x, str) else x)    
    
    # Get tokens by category
    tokens_by_category = get_tokens_by_category(data)
    
    # Remove stopwords from tokens
    tokens_by_category_stopwords_removed = remove_stopwords(tokens_by_category)
    
    # Take 5000 random tokens from each category for testing
    # for category, tokens in tokens_by_category_stopwords_removed.items():
    #     tokens_by_category_stopwords_removed[category] = tokens[:5000]
    
    # Initialize dictionaries to store corrections and distances
    corrections_dict = {}
    distances_dict = {}
    already_found = {}
    chunk_size = 500  # Chunk size for parallel processing
 
    # Calculate the total number of chunks for progress bar
    total_chunks = sum(math.ceil(len(tokens) / chunk_size) for tokens in tokens_by_category_stopwords_removed.values())
    
    # Use tqdm to display a progress bar so that my head doesn't explode while waiting for this to execute
    with tqdm(total=total_chunks, desc="Processing chunks of tokens") as pbar:
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor() as executor:
            futures = []

            # Submit each chunk as a separate task
            for category_name, tokens in tokens_by_category_stopwords_removed.items():
                for tokens_chunk in chunk_tokens(tokens, chunk_size):
                    future = executor.submit(find_corrections_and_distances, category_name, tokens_chunk, already_found)
                    futures.append(future)

            # Collect results from completed chunks and update dictionaries
            for future in as_completed(futures):
                category_name, corrections, distances, local_found = future.result()
                
                # Accumulate corrections and distances for each category
                if category_name not in corrections_dict:
                    corrections_dict[category_name] = []
                    distances_dict[category_name] = []
                
                corrections_dict[category_name].extend(corrections)
                distances_dict[category_name].extend(distances)
                
                # Update the already found dictionary
                already_found.update(local_found)

                # Update progress bar
                pbar.update(1)
        
        # Save corrections and distances to pickle files
        with open('corrections.pkl', 'wb') as f:
            pickle.dump(corrections_dict, f)
        with open('distances.pkl', 'wb') as f:
            pickle.dump(distances_dict, f)
        
    # Print statistics for each category (mean, median and standard deviation)
    for category, distances in distances_dict.items():
        print(f"Category: {category}")
        print(f"Mean: {np.mean(distances)}")
        print(f"Median: {np.median(distances)}")
        print(f"Standard deviation: {np.std(distances)}")
        print()
        
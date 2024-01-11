# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 18:21:41 2023

@author: bommi
"""

pip install pandas scikit-learn

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

book_data = pd.read_csv("book.csv",encoding="latin1")
book_data .dtypes

print(book_data.head())

# Assuming there's a 'title' column with book titles and a 'description' column with book descriptions
# You can adjust these column names based on your actual dataset structure
title_column = 'Book.Title'
description_column = 'Book.Rating'
# Ensure the ratings are in a numeric format
book_data['Book.Rating'] = pd.to_numeric(book_data['Book.Rating'], errors='coerce')

# Fill NaN values in the ratings column with the mean rating
book_data['Book.Rating'] = book_data['Book.Rating'].fillna(book_data['Book.Rating'].mean())

# Normalize ratings to a common scale (e.g., 0 to 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
book_data['Book.Rating'] = scaler.fit_transform(book_data['Book.Rating'].values.reshape(-1, 1))
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
book_data['Book.Rating'] = le.fit_transform(book_data['Book.Rating'].values.reshape(-1, 1))

# Create a dictionary to map book titles to integer indices
title_to_index = {title: idx for idx, title in enumerate(book_data[title_column].unique())}

# Create a sparse matrix for user-item ratings
from scipy.sparse import csr_matrix
ratings_matrix = csr_matrix((book_data["Book.Rating"], (book_data[title_column].map(title_to_index), book_data.index)))
#Create a sparse matrix for user-item ratings

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(ratings_matrix, ratings_matrix)

# Function to get book recommendations based on cosine similarity
def get_book_recommendations(book_title, cosine_sim_matrix, titles):
    if book_title not in title_to_index:
        print(f"Book '{book_title}' not found in the dataset.")
        return None
    
    idx = title_to_index[book_title]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar books (excluding itself)
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]

# Example: Get recommendations for a specific book
book_title_to_recommend = 'Example Book Title'
recommendations = get_book_recommendations(book_title_to_recommend, cosine_sim, book_data[title_column])

if recommendations is not None:
    print(f"Recommended books for '{book_title_to_recommend}':\n{recommendations}")
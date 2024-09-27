import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Step 1: Load the Datasets
# Replace these paths with the actual paths to your downloaded CSV files
books = pd.read_csv('Books.csv', low_memory=False)
users = pd.read_csv('Users.csv', low_memory=False)
ratings = pd.read_csv('Ratings.csv', low_memory=False)

# Step 2: Rename Columns for Consistency and Easier Access
books.columns = ['ISBN', 'Title', 'Author', 'Year', 'Publisher']
users.columns = ['User_ID', 'Age']
ratings.columns = ['User_ID', 'ISBN', 'Rating']

# Step 3: Drop Unnecessary Columns in Books DataFrame
books = books[['ISBN', 'Title', 'Author', 'Year', 'Publisher']]

# Ensure User_ID is of the same type in both DataFrames
ratings['User_ID'] = ratings['User_ID'].astype(str)
users['User_ID'] = users['User_ID'].astype(str)

# Step 4: Merge the Datasets to Create a Combined DataFrame
merged_df = pd.merge(ratings, books, on='ISBN', how='inner')
merged_df = pd.merge(merged_df, users, on='User_ID', how='inner')

# Display a sample of the merged DataFrame
print("\nMerged Data Sample:\n", merged_df.head())

# Step 5: Check and Handle Missing Values
print("\nMissing Values:\n", merged_df.isnull().sum())
merged_df.dropna(inplace=True)

# Step 6: Convert Data Types (Year and Rating)
merged_df['Year'] = pd.to_numeric(merged_df['Year'], errors='coerce')
merged_df['Rating'] = pd.to_numeric(merged_df['Rating'], errors='coerce')

# Step 7: Filter to Include Only Popular Books and Active Users
popular_books = merged_df.groupby('Title').filter(lambda x: x['Rating'].count() >= 20)
active_users = popular_books.groupby('User_ID').filter(lambda x: x['Rating'].count() >= 20)

# Display a sample of the popular books
print("\nPopular Books Sample:\n", popular_books.head())

# Step 8: Create a User-Item Matrix
user_item_matrix = popular_books.pivot_table(index='User_ID', columns='Title', values='Rating').fillna(0)

# Convert the user-item matrix to a sparse matrix
user_item_sparse = csr_matrix(user_item_matrix.values)

# Step 9: Calculate User Similarity Using Cosine Similarity
user_similarity = cosine_similarity(user_item_sparse, dense_output=False)

# Convert the sparse similarity matrix back to a DataFrame
user_similarity_df = pd.DataFrame.sparse.from_spmatrix(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Display the user similarity matrix sample
print("\nUser Similarity Matrix Sample:\n", user_similarity_df.head())

# Step 10: Build the Collaborative Filtering Recommendation Function
def get_book_recommendations(user_id, num_recommendations=5):
    if user_id not in user_similarity_df.index:
        print("User not found in the dataset.")
        return []

    # Find similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:num_recommendations + 1]

    # Collect books rated by similar users
    books_rated_by_similar_users = user_item_matrix.loc[similar_users].mean(axis=0)

    # Recommend books that the current user hasn't rated
    books_rated_by_user = user_item_matrix.loc[user_id]
    books_to_recommend = books_rated_by_similar_users[books_rated_by_user == 0].sort_values(ascending=False)

    # Get the book titles from the ISBNs
    return books_to_recommend.head(num_recommendations)

# Example: Get book recommendations for a specific user
user_id = '276747'
recommended_books = get_book_recommendations(user_id)

# Display the recommendations
print(f"\nTop book recommendations for User {user_id}:\n", recommended_books)

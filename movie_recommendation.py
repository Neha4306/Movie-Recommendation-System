import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Merge movies and ratings
data = pd.merge(ratings, movies, on="movieId")

# Create User-Movie Matrix
user_movie_matrix = data.pivot_table(
    index="userId",
    columns="title",
    values="rating"
)

# Replace missing values with 0
user_movie_matrix = user_movie_matrix.fillna(0)

# Calculate similarity between movies
movie_similarity = cosine_similarity(user_movie_matrix.T)

# Create similarity DataFrame
movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=user_movie_matrix.columns,
    columns=user_movie_matrix.columns
)

print("Movie similarity matrix created successfully!")

# Ask user for a movie
movie_name = input("\nEnter a movie name: ")

# Check if movie exists
if movie_name in movie_similarity_df.columns:

    similar_movies = movie_similarity_df[movie_name].sort_values(ascending=False)

    print("\n========== MOVIE RECOMMENDATION SYSTEM ==========")
    print(f"\nMovie Selected: {movie_name}")
    print("\nTop 5 Recommended Movies:\n")

    count = 1

    for movie in similar_movies.index[1:6]:
        print(f"{count}. {movie}")
        count += 1

else:
    print("\nMovie not found! Please enter a valid movie name.")
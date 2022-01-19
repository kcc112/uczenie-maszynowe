import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import math
import operator

# def square_rooted(x):
#   return round(math.sqrt(sum(a * a for a in x)), 3)

# def cosine_similarityy(x, y):
#   numerator = sum(a * b for a, b in zip(x,y))
#   denominator = square_rooted(x) * square_rooted(y)

#   return round(numerator / float(denominator), 3)

K = 8

def similar_users(user_id, matrix):
    # wybranie użytkownika
    user = matrix[matrix.index == user_id]

    # cała reszta
    other_users = matrix[matrix.index != user_id]

    # podobieństwo do reszty
    similarities = cosine_similarity(user, other_users)[0].tolist()

    # pary użytkownik podobieństwo
    indices = other_users.index.tolist()
    index_similarity = dict(zip(indices, similarities))

    # sortowanie na podstawie podobieństwo
    index_similarity_sorted = sorted(index_similarity.items(), key=operator.itemgetter(1))
    
    # wybranie k najbardziej podobnych użytkowników
    top_users_similarities = index_similarity_sorted[-K:]
    users = [u[0] for u in top_users_similarities]

    return users

def calculate_rating(row, similar_users, train):
    ratings = []

    # Wybranie użytkowników którzy ocenili dany film
    users_movie_ratings = train.loc[train['movie_id'] == row['movie_id'], ['user_id', 'rating']]

    for user_id in similar_users:
        value = users_movie_ratings.loc[(train['user_id'] == user_id), 'rating']
        if len(value) > 0:
            ratings.append(value.iloc[0])

    result = ratings[:K]

    if(len(result) == 0):
        return 3

    return sum(result) / len(result)

train = pd.read_csv('train.csv', sep=';', names=["id", "user_id", "movie_id", "rating"])
task = pd.read_csv('task.csv', sep=';', names=["id", "user_id", "movie_id", "rating"])

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot.html
rating_matrix = train.pivot(index='user_id', columns='movie_id', values='rating')

rating_matrix = rating_matrix.fillna(0)

# print(rating_matrix)

# new_df = train.pivot(index='movie_id',columns='user_id',values='rating')
# correlated_users = new_df.corr(method ='pearson')

# print(new_df)

new_task = task.copy()

for index, row in task.iterrows():
    print(str(index) + " / " + str(len(task.index)), end='\r')
    s_users = similar_users(row['user_id'], rating_matrix)
    score = calculate_rating(row, s_users, train)    

    task.loc[index, 'rating'] = str(int(round(score)))        

task.to_csv('submission.csv', sep=';', index=False, header=False)
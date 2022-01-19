import pandas as pd
import numpy as np

K = 5
N = 500
LEARNING_RATE = 0.001

train = pd.read_csv('train.csv', sep=';', names=["id", "user_id", "movie_id", "rating"])
task = pd.read_csv('task.csv', sep=';', names=["id", "user_id", "movie_id", "rating"])

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot.html
ratings_matrix = train.pivot(index='user_id',columns='movie_id',values='rating')

users_count, movies_count = ratings_matrix.shape

P = np.random.uniform(0, 1, [K, users_count])
X = np.random.uniform(0, 1, [K, movies_count])

ratings_matrix_numpy = ratings_matrix.to_numpy()

users, movies = np.where(np.isnan(ratings_matrix_numpy) == False)

N = len(list(zip(users, movies)))

for epoch in range(N):
    for user, movie in zip(users, movies):
        error = np.dot(P[:,user].T, X[:,movie]) - ratings_matrix_numpy[user, movie]
        error = error / K
        P[:, user] = P[:, user] - LEARNING_RATE * (error * X[:, movie])
        X[:, movie] = X[:, movie] - LEARNING_RATE * (error * P[:, user])

    predictions = np.dot(P.T, X)

    val_error = 0

    for user, movie in zip(users, movies):
      val_error += (ratings_matrix_numpy[user, movie] - predictions[user, movie]) ** 2

    print(f'\repoch {epoch + 1}/{N} error {val_error / (2 * N)}', end='\r')

predictions = np.dot(P.T, X)

print(ratings_matrix)

output = pd.DataFrame.from_records(predictions, columns=ratings_matrix.columns)    

print(output)

for index, row in task.iterrows():
  row_nr = ratings_matrix.index.get_loc(row['user_id'])
  movie_id = row['movie_id']
  score = output.loc[row_nr, movie_id]

  if score < 0:
    score = 0

  if score > 5:
    score = 5

  task.loc[index, 'rating'] = str(int(round(score)))   

task.to_csv("submission.csv", sep=';', index=False, header=False)
import argparse
import pandas as pd
from ast import literal_eval
import numpy as np
import operator

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--movies_file")
ap.add_argument("-t", "--task_file")
ap.add_argument("-z", "--train_file")
args = ap.parse_args()


def to_binary(values_list, unique_list):
    binary_list = []

    for unique in unique_list:
        if unique in values_list:
            binary_list.append(1)
        else:
            binary_list.append(0)

    return binary_list

def bool_to_num(value):
    if value == False:
      return 1
    else:
      return 0


# calculate hamming distance
# https://machinelearningmastery.com/distance-measures-for-machine-learning/
def hamming_distance(a, b):
  return sum(abs(e1 - e2) for e1, e2 in zip(a, b)) / len(a)


def distance(movies, movie_id_1, movie_id_2):
    movie1 = movies.iloc[movie_id_1 - 1]
    movie2 = movies.iloc[movie_id_2 - 1]

    genres_dist = hamming_distance(movie1['genres_binary'], movie2['genres_binary'])
    keywords_dist = hamming_distance(movie1['keywords_binary'], movie2['keywords_binary'])
    spoken_languages_dist = hamming_distance(movie1['spoken_languages_binary'], movie2['spoken_languages_binary'])

    budget_dist = abs(movie1['budget'] - movie2['budget'])
    popularity_dist = abs(movie1['popularity'] - movie2['popularity'])
    vote_average_dist = abs(movie1['vote_average'] - movie2['vote_average'])

    return genres_dist + keywords_dist + spoken_languages_dist + budget_dist + popularity_dist + vote_average_dist


# 1. Wczytanie danych
movies = pd.read_csv(args.movies_file, converters={
  "genres":literal_eval,
  "popularity":literal_eval,
  "spoken_languages":literal_eval,
  "vote_average":literal_eval,
  "keywords":literal_eval,
  "budget":literal_eval,
  "adult":literal_eval
  })

train = pd.read_csv(args.train_file, sep=';', names=["id", "user_id", "movie_id", "evaluation"])
task = pd.read_csv(args.task_file, sep=';', names=["id", "user_id", "movie_id", "evaluation"])

# 2. Obróbka danych
all_spoken_languages = []
all_keywords = []
all_genres = []

for index, row in movies.iterrows():
    for genre in row['genres']:
      if genre not in all_genres:
          all_genres.append(genre)

    for spoken_language in row['spoken_languages']:
        if spoken_language not in all_spoken_languages:
            all_spoken_languages.append(spoken_language)

    for keyword in row['keywords']:
        if keyword not in all_keywords:
            all_keywords.append(keyword)

movies['genres_binary'] = movies['genres'].apply(lambda x: to_binary(x, all_genres))
movies['spoken_languages_binary'] = movies['spoken_languages'].apply(lambda x: to_binary(x, all_spoken_languages))
movies['keywords_binary'] = movies['keywords'].apply(lambda x: to_binary(x, all_keywords))
movies['adult_num'] = movies['adult'].apply(lambda x: bool_to_num(x))

# https://stackoverflow.com/questions/56231450/ilocation-based-boolean-indexing-on-an-integer-type-is-not-available
# http://didactic.plantation.ics.p.lodz.pl/farm/uploads/executions/129/37/19692/19650/problem/web/index.html
task['evaluation'] = task['evaluation'].apply(lambda x: 0)

new_task = task.copy()

# print(task)

# http://didactic.plantation.ics.p.lodz.pl/farm/uploads/executions/129/37/19686/19644/problem/web/index.html
# 
K = 20 # k=4 best ? 

# 3. Knn
for nr, row in task.iterrows():
    # https://stackoverflow.com/questions/3002085/python-to-print-out-status-bar-and-percentage 
    print(str(nr) + " / " + str(len(task.index)), str(len(new_task.index)), end ='\r')

    distances = []
    k_neighbours = []

    trein_for_user = train.loc[train['user_id'] == row["user_id"]]

    for index, train_row in trein_for_user.iterrows():
        dist = distance(movies, row['movie_id'], train_row['movie_id'])
        distances.append((dist, train_row['movie_id'], train_row['evaluation']))

    # https://stackoverflow.com/questions/36955553/sorting-list-of-lists-by-the-first-element-of-each-sub-list
    distances.sort()

    for x in range(K):
        k_neighbours.append(distances[x])

    k_neighbours_df = pd.DataFrame(k_neighbours, columns=['dist', 'movie_id', 'evaluation'])
    new_task.loc[nr, 'evaluation'] = int(round(k_neighbours_df['evaluation'].mean()))

new_task.to_csv("submission.csv", sep=';', index=False, header=False)

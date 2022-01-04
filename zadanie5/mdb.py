import tmdbsimple as tmdb
import argparse
import pandas as pd

tmdb.API_KEY = ''

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--movie_file")
args = ap.parse_args()

movie_list = pd.read_csv(args.movie_file, sep=';', names=["nr", "movie_id", "title"])

movies = pd.DataFrame(columns=['popularity', 'vote_average', 'budget'])

for index, row in movie_list.iterrows():
    movie = tmdb.Movies(row['movie_id'])
    movie_info = movie.info()

    nr = row['nr']
    popularity = movie_info['popularity']
    vote_average = movie_info['vote_average']
    budget = movie_info['budget']
    movies.loc[nr - 1] = [popularity, vote_average, budget]

movies.to_csv('movies.csv', index=False)

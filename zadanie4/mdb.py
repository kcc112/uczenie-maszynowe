import tmdbsimple as tmdb
import argparse
import pandas as pd

tmdb.API_KEY = ''

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--movie_file")
args = ap.parse_args()

movie_list = pd.read_csv(args.movie_file, sep=';', names=["nr", "movie_id", "title"])

movies = pd.DataFrame(columns=['nr', 'genres', 'popularity', 'spoken_languages', 'vote_average', 'keywords', 'budget', 'adult'])

for index, row in movie_list.iterrows():
    movie = tmdb.Movies(row['movie_id'])
    movie_info = movie.info()
    movie_keywords = movie.keywords()

    nr = row['nr']
    genres = list(map(lambda genre: genre['id'], movie_info['genres']))
    popularity = movie_info['popularity']
    spoken_languages = list(map(lambda genre: genre['english_name'], movie_info['spoken_languages'])) # country english_name
    vote_average = movie_info['vote_average']
    keywords = list(map(lambda keyword: keyword['id'], movie_keywords['keywords']))
    budget = movie_info['budget']
    adult = movie_info['adult']
    movies.loc[nr - 1] = [nr, genres, popularity, spoken_languages, vote_average, keywords, budget, adult]

movies.to_csv('movies.csv', index=False)

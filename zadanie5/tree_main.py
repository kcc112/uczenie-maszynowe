from random import seed
from random import randrange
from csv import reader
import random
import pandas as pd
from anytree import Node, RenderTree
from anytree.exporter import DotExporter


def calculate_accuracy(actual, predicted):
  correct = 0

  for i in range(len(actual)):
    if actual[i] == predicted[i]:
      correct += 1
  return correct / float(len(actual)) * 100.0


def split_data_based_on_value(index, value, dataset):
  left = []
  right = []

  for row in dataset:
    if row[index] < value:
      left.append(row)
    else:
      right.append(row)

  return left, right


def gini_index(groups, classes):
  # suma wierszy w obu grupach można wydzielić bo zawsze będzie ta sama
  n_instances = float(sum([len(group) for group in groups]))
  gini = 0.0

  for group in groups:
    size = float(len(group))

    # perfect balance as all things should be
    if size == 0:
      continue

    score = 0.0

    for class_val in classes:
      p = [row[-1] for row in group].count(class_val) / size
      score += p * p
    gini += (1.0 - score) * (size / n_instances)

  return gini


def get_best_spilit(dataset):
  class_values = list(set(row[-1] for row in dataset))
  best_index, best_value, best_score, best_groups = None, None, 1, None

  for index in range(len(dataset[0]) - 1):
    for row in dataset:
      groups = split_data_based_on_value(index, row[index], dataset)
      gini = gini_index(groups, class_values)

      if gini < best_score:
        best_index, best_value, best_score, best_groups = index, row[index], gini, groups

  return {'index': best_index, 'value': best_value, 'groups': best_groups}


def check_movie_rating(group):
  ratings = [row[-1] for row in group]
  # return max(set(ratings), key=ratings.count)
  return int(round(sum(ratings) / len(ratings)))


def predict(tree, row):
  if row[tree['index']] < tree['value']:
    if type(tree['left']) is dict:
      return predict(tree['left'], row)
    else:
      return tree['left']
  else:
    if type(tree['right']) is dict:
      return predict(tree['right'], row)
    else:
      return tree['right']


def decode_features(number):
  if number == 0:
    return 'popularity'
  elif number == 1:
    return 'vote_average'
  elif number == 2:
    return 'budget'
  else:
    return 'nie ma takiego numeru'


def draw_tree(tree, parent_node, depth):
  if type(tree['left']) is dict:
    left_node = Node(f'if < {tree["left"]["value"]} left, {decode_features(tree["left"]["index"])}, depth: {depth} left', parent=parent_node)
    draw_tree(tree['left'], left_node, depth + 1)
  else:
    Node(f'rating: {tree["left"]}, depth: {depth} left', parent=parent_node)

  if type(tree['right']) is dict:
    right_node = Node(f'if < {tree["right"]["value"]} left, {decode_features(tree["right"]["index"])}, depth: {depth} right', parent=parent_node)
    draw_tree(tree['right'], right_node, depth + 1)
  else:
    Node(f'rating: {tree["right"]}, depth: {depth} right', parent=parent_node)

    return


def split(node, max_depth, min_size, depth):
  left, right = node['groups']
  del(node['groups'])

  # Sprawdz czy da sie jeszcze podzelić
  if left == [] or right == []:
    node['left'] = check_movie_rating(left + right)
    node['right'] = check_movie_rating(left + right)
    return

  # Max depth
  if depth >= max_depth:
    node['left'], node['right'] = check_movie_rating(left), check_movie_rating(right)
    return

  # Left
  if len(left) <= min_size:
    node['left'] = check_movie_rating(left)
  else:
    node['left'] = get_best_spilit(left)
    split(node['left'], max_depth, min_size, depth + 1)

  # Right
  if len(right) <= min_size:
    node['right'] = check_movie_rating(right)
  else:
    node['right'] = get_best_spilit(right)
    split(node['right'], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
  root = get_best_spilit(train)
  split(root, max_depth, min_size, 1)
  return root


K_FOLDS = 5
MAX_DEPTH = 5 # best 5
MIN_SIZE = 10


# 1. Wczytanie danych
task = pd.read_csv('task.csv', sep=';', names=['id', 'user_id', 'movie_id', 'evaluation'])
train = pd.read_csv('train.csv', sep=';', names=['id', 'user_id', 'movie_id', 'evaluation'])

filename = 'movies.csv'
file = open(filename, "rt")
lines = reader(file)
movies = list(lines)

# Zamiana string na float
for i in range(len(movies[0])):
  for row in movies:
    row[i] = float(row[i])

# https://stackoverflow.com/questions/56231450/ilocation-based-boolean-indexing-on-an-integer-type-is-not-available
# http://didactic.plantation.ics.p.lodz.pl/farm/uploads/executions/129/37/19692/19650/problem/web/index.html
task['evaluation'] = task['evaluation'].apply(lambda x: 0)

new_task = task.copy()

previous_user_id = None
previous_tree = None

for nr, task_row in task.iterrows():
  # https://stackoverflow.com/questions/3002085/python-to-print-out-status-bar-and-percentage 
  print(str(nr) + " / " + str(len(task.index)), end ='\r')

  # Wybranie danych testowych dla konkretnego użytkownika
  trein_for_user = train.loc[train['user_id'] == task_row["user_id"]]

  dataset = []
  scores = []
  folds = []

  # Wybranie filmów z danych testowych użytkownika
  for index, train_row in trein_for_user.iterrows():
    train_movie_info = movies[train_row['movie_id'] - 1].copy()
    train_movie_info.append(train_row['evaluation'])
    dataset.append(train_movie_info)

  # 3. Podzial na podzbiory
  # fold_size = int(len(dataset) / K_FOLDS)
  # dataset_copy = list(dataset.copy())

  # for i in range(K_FOLDS):
  #   fold = []

  #   while len(fold) < fold_size:
  #     index = random.randrange(len(dataset_copy))
  #     fold.append(dataset_copy.pop(index))
  #   folds.append(fold)

  # 4. Tworzenie najlepszego drzewa
  # for fold in folds:
  #   training_set = list(folds.copy())
  #   training_set.remove(fold)
  #   training_set = sum(training_set, [])
  #   validation_set = list(fold.copy())

  #   tree = build_tree(training_set, MAX_DEPTH, MIN_SIZE)
  #   predictions = []

  #   for row in validation_set:
  #     prediction = predict(tree, row)
  #     predictions.append(prediction)

  #   actual = [row[-1] for row in fold]
  #   accuracy = calculate_accuracy(actual, predictions)
  #   scores.append(accuracy)

  if task_row["user_id"] != previous_user_id:
    tree = build_tree(dataset, MAX_DEPTH, MIN_SIZE)
    previous_user_id = task_row["user_id"]

  new_task.loc[nr, 'evaluation'] = predict(tree, movies[task_row['movie_id'] - 1])

  # print('Scores: %s' % scores)
  # print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

  if task_row["user_id"] == 1642 and task_row["movie_id"] == 68:
    root = Node(f'if < {tree["value"]} left, {decode_features(tree["index"])}', id='root')
    draw_tree(tree, root, 1)
    DotExporter(root).to_picture("tree.png")

new_task.to_csv("submission.csv", sep=';', index=False, header=False)

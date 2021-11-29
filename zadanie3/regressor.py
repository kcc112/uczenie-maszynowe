import sys
import random
import argparse
import fileinput
import itertools

K_FOLDS = 4

CROSS_VALIDATION_EPOCHS = 250 # 250
CROSS_VALIDATION_LEARNING_RATE = 0.1

TRAINING_EPOCHS = 1000 # 1000
TRAINING_LEARNING_RATE = 0.01

DEGREES = [2, 3, 4, 5, 6, 7, 8, 9]


def normalize(value, min_value, max_value):
    result = (value - min_value) / (max_value - min_value)
    result = (2 * result) - 1
    return result


def denormalize(value, min_value, max_value):
    result = (value + 1) / 2
    result = result * (max_value - min_value) + min_value
    return result


def normalize_data(data, min_values, max_values):
    for i in range(len(data[0])):
        for j in range(len(data)):
            data[j][i] = normalize(data[j][i], min_values[i], max_values[i])

    return data

# https://stackoverflow.com/questions/7748442/generate-all-possible-lists-of-length-n-that-sum-to-s-in-python
def sums(length, total_sum):
    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in sums(length - 1, total_sum - value):
                yield (value,) + permutation

# https://machinelearningjourney.com/index.php/2020/02/08/polynomial-features/
def inputs_permutations(inputs, k):
    n = len(inputs[0])
    perms = []
    new_inputs = []

    for i in range(1, k):
        perms += list(sums(n, i))

    for input in inputs:
        new_input = []
        for perm in perms:
            var = 1
            for i in range(len(perm)):
                if(perm[i] != 0):
                    var *= input[i] ** perm[i]
            new_input.append(var)
        new_inputs.append(new_input)

    return new_inputs


def validate(inputs, expected, coef):
    summation = 0

    for i in range(len(inputs)):
        row = inputs[i]
        exp = expected[i]
        prediction = calculate(row, coef)
        summation = summation + pow(exp - prediction, 2)

    mse = summation / len(inputs)

    return mse


def calculate(row, coef):
    y = coef[0]
    
    for i in range(len(row) - 1):
        y += coef[i + 1] * row[i]

    return y


def get_coefficients(inputs, expected, coef, learning_rate):
    for i in range(len(inputs)):
        row = inputs[i]
        y = expected[i]

        prediction = calculate(row, coef)
        error = prediction - y
        coef[0] = coef[0] - learning_rate * error

        # new_w = w - detlaw
        # detlaw = error * input
        for i in range(len(row)-1):
            coef[i + 1] = coef[i + 1] - learning_rate * error * row[i]

    return coef


def train(training_set, validation_set, degree, learning_rate, epochs):
    train_target = []
    train_target_output = []
    validate_target = []
    validate_target_output = []

    for j in training_set.copy():
        train_target_output.append(j[-1])
        train_target.append(j[:-1])

    for j in validation_set.copy():
        validate_target_output.append(j[-1])
        validate_target.append(j[:-1])

    train_data = inputs_permutations(train_target, degree)
    validate_data = inputs_permutations(validate_target, degree)

    weights = [random.uniform(-1, 1) for i in range(len(train_data[0]) + 1)]

    validate_error = 0
    error_counter = 0
    last_validate_error = None
    best_validate_error = 0
    best_weights = weights

    for epoch in range(epochs):
        weights = get_coefficients(train_data, train_target_output, weights, learning_rate)
        validate_error = validate(validate_data, validate_target_output, weights)

        if last_validate_error is not None and validate_error > last_validate_error:
            error_counter += 1
        else:
            best_validate_error = validate_error
            error_counter = 0
            best_weights = weights

        if error_counter > 100:
            break
        last_validate_error = validate_error

    return best_validate_error, best_weights


def predict(inputs, weights, k):
    n = len(inputs)
    perms = []
    new_input = []

    for i in range(1, k):
        perms += list(sums(n, i))
    
    for perm in perms:
        var = 1
        for i in range(len(perm)):
            if(perm[i] != 0):
               var *= inputs[i] ** perm[i]
        new_input.append(var)

    return calculate(new_input, weights)

# 1. Wczytanie danych

train_data = []
test_data = []

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train_data")
args = ap.parse_args()

for line in fileinput.input(files=args.train_data):
    train_data.append([float(x) for x in line.split()])

for line in sys.stdin:
    test_data.append([float(x) for x in line.split()])

# 2. Normalizacja

min_values = []
max_values = []

for i in range(len(train_data[0])):
    min_values.append(min(train_data, key=lambda x: x[i])[i])
    max_values.append(max(train_data, key=lambda x: x[i])[i])

train_data = normalize_data(train_data, min_values, max_values)
test_data = normalize_data(test_data, min_values, max_values)

# 3. Podzial na podzbiory

folds = []
fold_size = int(len(train_data) / K_FOLDS)
train_data_copy = list(train_data.copy())

for i in range(K_FOLDS):
  fold = []
  
  while len(fold) < fold_size:
      index = random.randrange(len(train_data_copy))
      fold.append(train_data_copy.pop(index))
  folds.append(fold)

# 3. Szukanie najlepszego stopnia wielomianu

all_errors = []

for degree in DEGREES:
    error = 0

    for fold in folds:
        training_set = list(folds.copy())
        training_set.remove(fold)
        training_set = sum(training_set, [])
        validation_set = list(fold.copy())

        fold_error, _ = train(training_set, validation_set, degree, CROSS_VALIDATION_LEARNING_RATE, CROSS_VALIDATION_EPOCHS)
        error += fold_error

    error /= K_FOLDS
    all_errors.append(error)

index = all_errors.index(min(all_errors))
degree = DEGREES[int(index)]

# print(degree)

# 4. Trenowanie na znalezionym stopniu

_, best_weights = train(train_data, train_data, degree, TRAINING_LEARNING_RATE, TRAINING_EPOCHS)

# print(min_values, max_values)

for x in test_data:
    result = predict(x, best_weights, degree)
    print(denormalize(result, min_values[-1], max_values[-1]))

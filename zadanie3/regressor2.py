import fileinput
import argparse
import sys
import random

FOLDS_NUM = 4
POLYNOMIAL_DEGREES = [1, 2, 3, 4, 5, 6, 7, 8]
EPOCHS_NUM = 20
LEARNING_RATE = 0.3

#################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-t','--train_set')
args = parser.parse_args()

training_data = []
testing_data = []

for line in fileinput.input(files=args.train_set):
    training_data.append([float(x) for x in line.split()])

for line in sys.stdin:
    testing_data.append([float(x) for x in line.split()])

################################################################

folds = []
fold_size = int(len(training_data) / FOLDS_NUM)
training_data_copy = list(training_data.copy())

for i in range(FOLDS_NUM):
  fold = []
  
  while len(fold) < fold_size:
      index = random.randrange(len(training_data_copy))
      fold.append(training_data_copy.pop(index))
  folds.append(fold)

################################################################

def divide_dataset(dataset):
  target = []
  target_output = []    

  for row in dataset.copy():
      target_output.append(row[-1])
      target.append(row[:-1])

  return target, target_output

def calculate_f(x, p):
  result = 0
  size = len(p) - 1

  for z in range(1, size):
      f_p = p[z][1]
      f_p_x = int(p[z][0]) # index x1 x2 x3
      result += x[f_p_x] * f_p

  result += p[-1][1] # wyraz wolny [0, 1.0]

  return result

def calculate_f2(x, p):
  # print(x, 'x')
  # print(p, 'p')
  result = 0
  size = len(p) - 1

  for z in range(1, size):
    f_p = p[z][1]
    f_p_x = int(p[z][0])
    # print(f_p_x, 'f_p_x')
    result += pow(x[f_p_x], int(p[z][0])) * f_p

  result += p[-1][1]

  # print(result, 'result')

  return 0

def calculate_f3(x, p, polynomial_degree):
#   print(x, 'kamil', polynomial_degree)
#   print(p, 'p')
  result = 0
  size = len(p) - 1

  for z in range(1, size):
    f_p = p[z][1]
    f_p_x = int(p[z][0])
    # print(x, 'f_p_x')
    result += pow(x[polynomial_degree], int(p[z][0])) * f_p

  result += p[-1][1]

#   print(result, 'result')

  return result


def create_input2(inputs, k):
    new_inputs = []

    # print(inputs, k, 'c')
    
    if k != 1:
      for input in inputs:
        new_input = list(input.copy())
        # print(new_input, 'z')
        
        for p in range(2, k + 1):
          for x in input:
            # print(x, p, 'pow')
            new_input.append(pow(x, p))

        new_input.insert(0, 1)
        # new_input.append(1)
        new_inputs.append(new_input)
    else:
      for input in inputs:
        new_input = list(input.copy())
        new_input.insert(0, 1)
        # new_input.append(1)
        new_inputs.append(new_input)

    return new_inputs

def validate(validate_data2, validate_target_output, description_in, polynomial_degree):
    # print(description_in)
    N = len(validate_data2)
    n = len(description_in) - 1
    errors_tmp = 0
    error = 0

    for p in range(n):
      for j in range(N):
        x = validate_data2[j]
        y = validate_target_output[j]

        errors_tmp += pow((calculate_f3(x, description_in, polynomial_degree) - y), 2)

    error = errors_tmp / (2 * N)

    # print(calculate_f3(x, description_in), y)

    return error



def cross_validation(training_set, validation_set, polynomial_degree):
    train_target, train_target_output = divide_dataset(training_set) 
    validate_target, validate_target_output = divide_dataset(validation_set)

    train_data = generate_inputs(train_target, polynomial_degree + 1)
    validate_data = generate_inputs(validate_target, polynomial_degree + 1)
    train_data2 = create_input2(train_target, polynomial_degree)
    validate_data2 = create_input2(validate_target, polynomial_degree)
    # print(validate_data, polynomial_degree, 'x1')
    # print(validate_data2, polynomial_degree, 'x3')
    # print(validate_data, polynomial_degree, 'x2')

    # print(train_data, polynomial_degree, 'x1')
    # print(train_data2, polynomial_degree, 'x3')

    # print(train_data2)
    # print('xd')
    # print(train_data)

    description_in = []

    description_in.append([1, 1])

    for i in range(len(train_data2[0]) - 1):
      description_in.append([float(i + 1), 1.0])

    description_in.append([0.0, 1.0])

    # print(description_in)

    count = 0
    error = 0
    errors = []
    stop_flag = False
    max_iterations = 300
    n = len(description_in) - 1
    N = len(train_data2)
    learning_rate = 0.1
    stop = 0.001

    # print(train_data2, 'xd1')
    # print(train_target_output, 'xd2')

    while max_iterations > 0 and stop_flag == False:
        stop_count = 0
        count += 1
        max_iterations -= 1
        gradients = [0] * n
        errors_tmp = 0

        for p in range(n):
            for j in range(N):
                x = train_data2[j]
                y = train_target_output[j]
      
                # print(description_in[n - p][0])
                gradients[p] += (calculate_f(x, description_in) - y) * train_data2[j][int(description_in[n - p][0])]
                errors_tmp += pow((calculate_f(x, description_in) - y), 2)

            gradients[p] = gradients[p] / N
            description_in[n - p][1] = description_in[n - p][1] - (learning_rate * gradients[p])

        error = errors_tmp / (2 * N)

        errors.append(error)

    validation_error = validate(validate_data2, validate_target_output, description_in, polynomial_degree)
    # print(validation_error)
    # print(errors[-1], polynomial_degree)

    return validation_error

####################################################################################################

def calculate_min_max(data):
    min_values, max_values = data[0], data[0] 
    
    for row in data[1:]:
        max_values = [max(x, y) for x, y in zip(max_values, row)]
        min_values = [min(x, y) for x, y in zip(min_values, row)]

    return min_values, max_values


def normalize_value(value, min_value, max_value):
    result = (value - min_value) / (max_value - min_value)
    result = (2 * result) - 1
    return result


def denormalize_value(value, min_value, max_value):
    result = (value + 1) / 2
    result = result * (max_value - min_value) + min_value
    return result


def normalize_data(data, min_values, max_values):
    for i in range(len(data[0])):
        for j in range(len(data)):
            data[j][i] = normalize_value(data[j][i], min_values[i], max_values[i])

    return data


def generate_folds(data):
    dataset_split = list()
    dataset_copy = list(data)
    fold_size = int(len(data) / FOLDS_NUM)
    for i in range(FOLDS_NUM):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def sums(length, total_sum):
    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in sums(length - 1, total_sum - value):
                yield (value,) + permutation


def create_permutations(n, k):
    perms = []
    for i in range(1, k):
        perms += list(sums(n, i))
    return perms


def create_input(perms, inputs):
    new_input = []
    for perm in perms:
        val = 1
        for i in range(len(perm)):
            if(perm[i] != 0):
                val *= inputs[i] ** perm[i]
        new_input.append(val)
    new_input.append(1)
    return new_input


def calculate(inputs, weights, k):
    perms = create_permutations(len(inputs), k)
    new_input = create_input(perms, inputs)    
    return vector_dot(new_input, weights)


def generate_inputs(inputs, k):  
    perms = create_permutations(len(inputs[0]), k)

    # print(inputs, 'xd1')
    
    new_inputs = []
    for input in inputs:
        new_input = create_input(perms, input)          
        new_inputs.append(new_input)

    # print(new_inputs, 'xd2')
    return new_inputs


def divide_dataset(dataset):
    target = []
    target_output = []    

    for row in dataset.copy():
        target_output.append(row[-1])
        target.append(row[:-1])

    return target, target_output


# def cross_validation2(training_set, validation_set, polynomial_degree):
#     train_target, train_target_output = divide_dataset(training_set) 
#     validate_target, validate_target_output = divide_dataset(validation_set)   

#     train_data = generate_inputs(train_target, polynomial_degree)
#     validate_data = generate_inputs(validate_target, polynomial_degree)

#     weights = [random.uniform(-1, 1) for col in range(len(train_data[0]))]

#     validation_cost = None
#     prev_cost = None
#     best_cost = None

#     for epoch in range(EPOCHS_NUM):        
#         train_prediction = matrix_dot(train_data, weights)
#         train_loss = vector_substract(train_prediction, train_target_output)
#         gradient = matrix_dot(matrix_transpose(train_data), train_loss)
#         weights = vector_substract(weights, vector_multiply(gradient, LEARNING_RATE / len(train_target_output)))

#         validation_prediction = matrix_dot(validate_data, weights)
#         validation_loss = vector_substract(validation_prediction, validate_target_output)
#         validation_cost = sum(vector_power(validation_loss, 2)) / len(validate_target_output)

#         if prev_cost is None or validation_cost < prev_cost:          
#             best_cost = validation_cost 
       
#         prev_cost = validation_cost

#     return best_cost

###################################################################################################

################################################################

def full_training(training_data, polynomial_degree):
    train_target, train_target_output = divide_dataset(training_data) 

    train_data2 = create_input2(train_target, polynomial_degree)

    description_in = []

    description_in.append([1, 1])

    for i in range(len(train_data2[0]) - 1):
      description_in.append([float(i + 1), 1.0])

    description_in.append([0.0, 1.0])

    count = 0
    error = 0
    errors = []
    stop_flag = False
    max_iterations = 250
    n = len(description_in) - 1
    N = len(train_data2)
    learning_rate = 0.3
    stop = 0.01

    # print(description_in)
    # print(train_data2)

    while max_iterations > 0 and stop_flag == False:
        stop_count = 0
        count += 1
        max_iterations -= 1
        gradients = [0] * n
        errors_tmp = 0

        for p in range(n):
            for j in range(N):
                x = train_data2[j]
                y = train_target_output[j]
      
                gradients[p] += (calculate_f(x, description_in) - y) * train_data2[j][int(description_in[n - p][0])]
                errors_tmp += pow((calculate_f(x, description_in) - y), 2)

            gradients[p] = gradients[p] / N

            description_in[n - p][1] = description_in[n - p][1] - (learning_rate * gradients[p])

        error = errors_tmp / (2 * N)

        # if len(errors) > 1 and errors[-1] - error <= stop:
        #     # print(error)
        #     stop_flag = True

        errors.append(error)

    # print(description_in)

    return description_in

################################################################


def normalize_value(value, min_value, max_value):
    result = (value - min_value) / (max_value - min_value)
    result = (2 * result) - 1
    return result


def denormalize_value(value, min_value, max_value):
    result = (value + 1) / 2
    result = result * (max_value - min_value) + min_value
    return result


def normalize_data(data, min_values, max_values):
    for i in range(len(data[0])):
        for j in range(len(data)):
            data[j][i] = normalize_value(data[j][i], min_values[i], max_values[i])

    return data

min_values, max_values = calculate_min_max(training_data)

training_data = normalize_data(training_data, min_values, max_values)
testing_data = normalize_data(testing_data, min_values, max_values)

all_errors = []

k = 1

for polynomial_degree in POLYNOMIAL_DEGREES:
    error = 0

    for fold in folds:
        training_set = list(folds.copy())
        training_set.remove(fold)
        training_set = sum(training_set, [])

        validation_set = list(fold.copy())

        # print(cross_validation(training_set, validation_set, polynomial_degree), 'my')
        error += cross_validation(training_set, validation_set, polynomial_degree)
        # print(cross_validation2(training_set, validation_set, polynomial_degree), 'other')

    # print(error / FOLDS_NUM)

    error /= FOLDS_NUM

    all_errors.append(error)

    # print(all_errors)

    index = all_errors.index(min(all_errors))

    k = POLYNOMIAL_DEGREES[int(index)]

    # print(k)

# print(all_errors)
# print(k)

best_weights = full_training(training_data, k)

# print(best_weights)

for x in testing_data:
  val = x
  val.insert(0, 1)
  y = calculate_f3(val, best_weights, k)
  print(denormalize_value(y, min_values[-1], max_values[-1]))

# print(y)


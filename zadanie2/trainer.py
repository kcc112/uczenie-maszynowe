import fileinput
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument('-t','--train_set')
parser.add_argument('-i','--data_in')
parser.add_argument('-o','--data_out')
args = parser.parse_args()


# odczytanie pliku description.txt i przerobienie na matrix 2

description_in = []

for line in sys.stdin:
    numbers = [float(number) for number in line.split(' ')]
    description_in.append(numbers)

# 1 1
# 1 1.0
# 0 1.0
# print(description_in_array)
# [[1.0, 1.0], [1.0, 1.0], [0.0, 1.0]]

#############################################################

# odczytanie pliku train_set.txt i przerobienie na matrix 2

train_set = []

for line in fileinput.input(files=args.train_set):
    numbers = [float(number) for number in line.split(' ')]
    numbers.insert(0,1)
    train_set.append(numbers)

# -2.96 -9.41
# print(train_set)
# [[1, -2.963146353518017, -9.417057387429033], [1, -2.927883513943802, -9.251066001047633], ...

############################################################

# odczytanie iteracji z pliku data_in.txt

max_iterations = 0

for line in fileinput.input(files=args.data_in):
    max_iterations = int(line.replace('iterations=', ''))

# print(max_iterations)

############################################################

# dotatkowe parametry

N = len(train_set)

# wymiar wielomianu + 1
n = len(description_in) - 1

# 0.3 best
learning_rate = 0.3

p_factors = []

# print(description_in)
# 1 1
# 1 1.0
# 0 1.0

for number in range(n):
    p_factors.append(float(description_in[number + 1][1]))

# print(p_factors)
# [1.0, 1.0]

###########################################################

# funkcjie

# x = [1, -2.963146353518017]
# p = [[1, 1], [1, 1.0], [0, 1.0]]
def calculate_f(x, p):
    result = 0
    size = len(p) - 1

    for z in range(1, size):
        f_p = p[z][1]
        f_p_x = int(p[z][0]) # index x1 x2 x3
        result += x[f_p_x] * f_p

    result += p[-1][1] # wyraz wolny [0, 1.0]

    return result

#########################################################3#

# Wylicz nowy description_in

count = 0

error = 0

errors = []

stop_flag = False

# max_iterations = 200

# 0.0005 best
stop = 0.001

while max_iterations > 0 and stop_flag == False:
    count += 1
    max_iterations -= 1

    # empty gradients array
    gradients = [0] * n

    errors_tmp = 0

    for p in range(n):
        # print(n, p)
        # 2 0 | n - p = 2
        # 2 1 | n - p = 1

        # len(p_factors) = 2 | len(description_in) = 3
        # p_factors[n - p - 1] = description_in[n - p][1] # X

        for j in range(N):

            # [1, -2.963146353518017, -9.417057387429033]
            # [1, -2.963146353518017]
            # input
            x = train_set[j][:-1]

            # desired output
            # -9.417057387429033
            y = train_set[j][-1]

            # (f(xj,p) - yj) * xij | jesli pierwsza iteracja to xij = 1
            # print(description_in[n - p][0])
            gradients[p] += (calculate_f(x, description_in) - y) * train_set[j][int(description_in[n - p][0])]

            errors_tmp += pow((calculate_f(x, description_in) - y), 2)

        # sum(gradient) / N   
        gradients[p] = gradients[p] / N

        # replace p(t + 1) = p(t) - learning_rate * gradient
        description_in[n - p][1] = description_in[n - p][1] - (learning_rate * gradients[p])

    error = errors_tmp / (2 * N)

    # print(errors)

    if len(errors) > 1 and errors[-1] - error <= stop:
        stop_flag = True

    errors.append(error)


##########################################################

# zapisz nowy description_in do description_out.txt

for index, array in enumerate(description_in):
    if index == 0:
        print(str(int(array[0])) + ' ' + str(int(array[1])))
    else:
        print(str(int(array[0])) + ' ' + str(array[1]))

########################################################

# zapisz ilosc wykonanych iteracji

f = open(args.data_out, "w")
f.write('iterations=' + str(count))
f.close()
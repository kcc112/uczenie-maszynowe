import fileinput
import argparse
import sys

# Read args
parser = argparse.ArgumentParser()
parser.add_argument('-t','--train_set')
parser.add_argument('-i','--data_in')
parser.add_argument('-o','--data_out')
args = parser.parse_args()

# Parse description_in.txt file
description_in_array = []

for line in sys.stdin:
    numbers = [float(number) for number in line.split(' ')]
    description_in_array.append(numbers)

expected_output = []
train_input = [[] for i in range(int(description_in_array[0][0]))]

for line in fileinput.input(files=args.train_set):
    numbers = [float(number) for number in line.split(' ')]
    for index, n in enumerate(numbers):
        if index != len(numbers) - 1:
            train_input[index].append(n)
    expected_output.append(numbers[-1])

for line in fileinput.input(files=args.data_in):
    iterations = int(line.replace('iterations=', ''))

# print(description_in_array)
# print(expected_output)
# print(train_input)
# print(iterations)

error = 0
errors = []
lerning_rate = 0.1
stop_arry = []
stop = 0
stop_flag = False
i = 0

while iterations > 0 and stop_flag == False:

    iterations -= 1

    if i > 0:
        stop = sum([pow(err, 2) for err in errors]) / len(errors)
        stop_arry.append(stop)

    # stop
    if i > 2 and stop_arry[len(stop_arry) - 2] - stop <= 0.0001:
        # print(i, stop, stop_arry, stop_arry[len(stop_arry) - 2] - stop)
        stop_flag = True
    
    i += 1

    # f(xj,p)
    outputs_1 = []

    # (f(xj,p) - yj)
    outputs_2 = []  

    for index, line in enumerate(fileinput.input(files=args.train_set)):
        train_set = [float(number) for number in line.split(' ')]
        # print(train_set)
        # print('line', line)
        output = 0

        for line_number, input in enumerate(description_in_array):
            array = []

            if line_number != 0:
                tmp = 1
            
                # print('input',input)
                for index, value in enumerate(input):
                    array_len = len(input)
                    if index < array_len and index != array_len - 1 and int(value) - 1 >= 0:
                        # print(train_set, train_set[line_number - 1], index)
                        tmp *= train_set[line_number - 1]
                    elif index == array_len - 1:
                        # print(value)
                        tmp *= value

                output += tmp
        
        outputs_1.append(output)
        # print(outputs_1)

    #     print(train_set)
    # print(description_in_array)
    # print(outputs_1)

    for index, output in enumerate(outputs_1):
        # print(output,  expected_output[index])
        outputs_2.append(output - expected_output[index])

    # print('outputs_2', outputs_2)

    # print([out_2 * x for out_2, x in zip(outputs_2, train_input)])

    error = sum([pow(out_2, 2) for out_2 in outputs_2]) / (2 * len(outputs_2))
    errors.append(error)

    # print('error', error)

    # print(outputs_2, train_input)

    for line_number, input in enumerate(description_in_array):
        array = []

        # print(input, line_number)
        # print(description_in_array)

        if line_number != 0:
            tmp = 1
            p = 1

            if line_number != len(description_in_array) - 1:
                # print(outputs_2, train_input[line_number - 1])
                # print('XD1',[out_2 * x for out_2, x in zip(outputs_2, train_input[line_number - 1])])
                # print('XD2',sum([out_2 * x for out_2, x in zip(outputs_2, train_input[line_number - 1])]))
                # print('XD3',sum([out_2 * x for out_2, x in zip(outputs_2, train_input[line_number - 1])]) / (len(description_in_array) - 1))
                # print('XD4',lerning_rate * sum([out_2 * x for out_2, x in zip(outputs_2, train_input[line_number - 1])]) / (len(description_in_array) - 1))
                # print('XD5',input[-1] - lerning_rate * sum([out_2 * x for out_2, x in zip(outputs_2, train_input[line_number - 1])]) / (len(description_in_array) - 1))


                p = input[-1] - (lerning_rate * (sum([out_2 * x for out_2, x in zip(outputs_2, train_input[line_number - 1])]) / (len(outputs_2))))
                # print(p)
            else:
                p = input[-1] - (lerning_rate * (sum(outputs_2) / (len(outputs_2))))
                # print(p)
            
            input[-1] = p

        # print(description_in_array)


for index, array in enumerate(description_in_array):
    if index != 0:
        print(str(array).replace(',', '').replace('[', '').replace(']', ''))
    else:
        print(str(int(array[0])) + ' ' + str(int(array[1])))

f = open(args.data_out, "w")
f.write('iterations=' + str(i))
f.close()
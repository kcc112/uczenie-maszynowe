import fileinput
import argparse
import sys

# Read args
parser = argparse.ArgumentParser()
parser.add_argument('-d','--description')
args = parser.parse_args()

# Parse in.txt file
inputs_array = []

for line in sys.stdin:
    numbers = [float(number) for number in line.split(' ')]
    inputs_array.append(numbers)

# print(inputs_array)

# Calculate
for input in inputs_array:
    output = 0
    array = []

    for line_number, line in enumerate(fileinput.input(files=args.description)):
        array = [float(number) for number in line.split(' ')]
        array_len = len(array)

        if line_number != 0:
            tmp = 1

            for index, value in enumerate(array):
                if index < array_len and index != array_len - 1 and int(value) - 1 >= 0:
                    tmp *= input[int(value) - 1]
                    # print(input[int(value) - 1])
                elif index == array_len - 1:
                    # print(value)
                    tmp *= value
            # print(tmp, array)S
            output += tmp

    print(output)

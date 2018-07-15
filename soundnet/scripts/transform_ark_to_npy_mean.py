import sys
import os
import numpy as np
import mmap
from tqdm import tqdm

if len(sys.argv) != 3:
    print("usage: {} [input ark file] [output dir]".format(sys.argv[0]))
    exit()

input_ark_filepath = sys.argv[1]
output_dirpath = sys.argv[2]

if not os.path.isdir(output_dirpath):
    os.makedirs(output_dirpath)

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1

    return lines

def write_npy(filename, array):
    array = np.asarray(array)
    np.save("{}/{}.npy".format(output_dirpath, filename), np.mean(array, axis=0))

def main():
    input_ark_file = open(input_ark_filepath, 'r')
    array = []
    for line in tqdm(input_ark_file, total=get_num_lines(input_ark_filepath), ncols=60):
        line = line.rstrip('\n')
        if '[' in line:
            # Begin of new file
            filename = line.split()[0]
            array = []
        elif ']' in line:
            # End of file
            vector = [float(i) for i in line.split()[:-1]]
            array.append(vector)
            write_npy(filename, array)
        else:
            # In the middle of file
            vector = [float(i) for i in line.split()]
            array.append(vector)

if __name__ == "__main__":
    main()

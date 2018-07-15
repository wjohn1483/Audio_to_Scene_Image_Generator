import sys
from tqdm import tqdm

if len(sys.argv) != 3:
    print("Usage: python transform_number_to_path.py [input file path] [output file path]")
    exit()

path_prefix = "mp3/videos/"
f = open(sys.argv[1], 'r')
output_file = open(sys.argv[2], 'w')

for line in tqdm(f):
    line = line.rstrip('\n')
    output_file.write(path_prefix + "{}/{}/{}/".format(line[-3], line[-2], line[-1]) + line + ".mp4.mp3\n")


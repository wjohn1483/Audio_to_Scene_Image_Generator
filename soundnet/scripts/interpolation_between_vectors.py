import sys
import numpy as np

if len(sys.argv) != 4:
    print("usage: python {} [source vector] [target vector] [output dir]".format(sys.argv[0]))
    exit()

source_vector_path = sys.argv[1]
target_vector_path = sys.argv[2]
output_dir = sys.argv[3]
num_of_step_to_target = 4

source_vector = np.load(source_vector_path)
target_vector = np.load(target_vector_path)

difference_between_vector_per_step = (target_vector - source_vector) / num_of_step_to_target

for i in range(0, num_of_step_to_target+1):
    np.save(output_dir + "/{}.npy".format(i), source_vector + i * difference_between_vector_per_step)

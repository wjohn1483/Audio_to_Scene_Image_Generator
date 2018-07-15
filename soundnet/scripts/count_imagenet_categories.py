import os
import collections
from tqdm import tqdm

result_filepath = "./all_images/results_all.txt"

# Make categories set
places_and_objects = collections.defaultdict(int)

# Count
f = open(result_filepath, 'r')
for line in tqdm(f, ncols=50):
    classified_result_per_image = line.split('[')[1][:-2]
    classified_result_per_image = classified_result_per_image.split("',")
    for name in classified_result_per_image:
        name = name.strip().strip("'").replace(" ","").split(",")
        for n in name:
            places_and_objects[n] += 1

# Sort and print counting result
places_and_objects = sorted(places_and_objects.items(), key=lambda k_v: k_v[1], reverse=True)
for i in range(len(places_and_objects)):
    print("{}\t{}".format(places_and_objects[i][0], places_and_objects[i][1]))


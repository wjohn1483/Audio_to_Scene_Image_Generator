import os
import collections

prediction_dir = "./results/"
predictions = os.listdir(prediction_dir)

# Make categories set
places = collections.defaultdict(int)
objects = collections.defaultdict(int)

# Count
for filename in predictions:
    f = open(prediction_dir + filename)
    for i, line in enumerate(f):
        place = line.split("Scene")[1].split(' ')[1].rstrip('\t').rstrip('\n')
        places[place] += 1
        objects_in_line = line.split("Scene")[0].split("Object")[1].split(' ')
        for obj in range(2, len(objects_in_line)-2):
            objects[objects_in_line[obj].rstrip(',')] += 1

    if i != (6000 - 1):
        print("{} failed, need to be extracted again".format(filename))

# Print places and numbers
print("\nScenes")
for key, value in places.items():
    print("{}\t{}".format(key, value))

# Print objects and numbers
print("\nObjects")
for key, value in objects.items():
    print("{}\t{}".format(key, value))


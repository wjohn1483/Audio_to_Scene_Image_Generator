from tqdm import tqdm

f = open("./mp3_public.list", 'r')
training_data_file = open("./training_data_list.txt", 'w')
testing_data_file = open("./testing_data_list.txt", 'w')

num_of_training_data = 6000
num_of_testing_data = 500
for i, line in tqdm(enumerate(f), ncols=60):
    if i < num_of_training_data:
        training_data_file.write(line)
    elif num_of_training_data <= i and i < num_of_training_data + num_of_testing_data:
        testing_data_file.write(line)
    if num_of_training_data + num_of_testing_data <= i:
        break


import os

# define the directory containing the text files
directory = 'data_object/training/label_2/'

first_words = set()

for filename in os.listdir(directory):
    if filename.endswith('.txt'):  # only .txt files
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            for line in file.readlines():
                first_word = line.split()[0]
                if len(first_word) > 0:
                    first_words.add(first_word)

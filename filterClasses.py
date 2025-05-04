import os

# 定义目标目录
directory = 'data_object/training/label_2/'

# 用于存储第一行第一个单词的集合
first_words = set()

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    if filename.endswith('.txt'):  # 只处理 .txt 文件
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            for line in file.readlines():
                # 获取第一行的第一个单词
                first_word = line.split()[0]
                if len(first_word) > 0:
                    first_words.add(first_word)

# 输出结果
print("第一行的第一个单词（去重）：", first_words)
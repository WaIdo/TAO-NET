# import json

# # 可修改的参数
# FILE_PATH_TRAIN = 'processed_train_data.json'  # 训练数据文件路径
# FILE_PATH_VAL = 'processed_val_data.json'      # 验证数据文件路径
# FILE_PATH_TEST = 'processed_test_data.json'    # 测试数据文件路径
# NUM_LINES = 5                                  # 要查看的行数
# FROM_END = False                               # 是否从文件末尾查看，True 为从文件末尾查看

# def read_and_print_json(file_path, num_lines=None, from_end=False):
#     """
#     读取并打印指定 JSON 文件的内容
#     参数:
#     file_path (str): JSON 文件路径
#     num_lines (int, optional): 要查看的行数，默认为 None（查看所有行）
#     from_end (bool, optional): 是否从文件末尾查看，默认为 False（从文件开头查看）
#     """
#     with open(file_path, 'r') as f:
#         data = json.load(f)
        
#         if num_lines is not None:
#             if from_end:
#                 data = data[-num_lines:]
#             else:
#                 data = data[:num_lines]
        
#         for entry in data:
#             print(json.dumps(entry, ensure_ascii=False, indent=2))

# # 查看训练数据
# print("训练数据内容:")
# read_and_print_json(FILE_PATH_TRAIN, num_lines=NUM_LINES, from_end=FROM_END)

# # 查看验证数据
# print("\n验证数据内容:")
# read_and_print_json(FILE_PATH_VAL, num_lines=NUM_LINES, from_end=FROM_END)

# # 查看测试数据
# print("\n测试数据内容:")
# read_and_print_json(FILE_PATH_TEST, num_lines=NUM_LINES, from_end=FROM_END)


import json

# 可修改的参数
FILE_PATH_TRAIN = 'processed_train.json'  # 训练数据文件路径
FILE_PATH_TEST = 'processed_test.json'    # 测试数据文件路径
FILE_PATH_VALID = 'processed_valid.json'  # 验证数据文件路径
NUM_LINES = 10                                  # 要查看的行数
FROM_END = False                               # 是否从文件末尾查看,True 为从文件末尾查看

def read_and_print_json(file_path, num_lines=None, from_end=False):
    """
    读取并打印指定 JSON 文件的内容
    参数:
    file_path (str): JSON 文件路径
    num_lines (int, optional): 要查看的行数,默认为 None（查看所有行）
    from_end (bool, optional): 是否从文件末尾查看,默认为 False（从文件开头查看）
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        if num_lines is not None:
            if from_end:
                lines = lines[-num_lines:]
            else:
                lines = lines[:num_lines]
        
        for line in lines:
            entry = json.loads(line)
            print(json.dumps(entry, ensure_ascii=False, indent=2))

# 分隔符函数
def print_separator():
    print("\n" + "=" * 40 + "\n")

# 查看训练数据
print("训练数据内容:")
read_and_print_json(FILE_PATH_TRAIN, num_lines=NUM_LINES, from_end=FROM_END)

# 打印分隔符
print_separator()

# 查看验证数据
print("验证数据内容:")
read_and_print_json(FILE_PATH_VALID, num_lines=NUM_LINES, from_end=FROM_END)

# 打印分隔符
print_separator()

# 查看测试数据
print("测试数据内容:")
read_and_print_json(FILE_PATH_TEST, num_lines=NUM_LINES, from_end=FROM_END)

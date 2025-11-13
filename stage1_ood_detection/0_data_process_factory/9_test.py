import json

# 使用 json.loads(line) 直接读取
def test_loads_direct(file_path):
    try:
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
        print("Using json.loads(line) worked correctly.")
        return data
    except Exception as e:
        print(f"Using json.loads(line) encountered an error: {e}")
        return None

# 使用 json.loads(line) 读取所有行后再解析
def test_loads_readlines(file_path):
    try:
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f.readlines()]
        print("Using json.loads(line) with readlines() worked correctly.")
        return data
    except Exception as e:
        print(f"Using json.loads(line) with readlines() encountered an error: {e}")
        return None

# 文件路径
file_path = 'sampled_train_data.json'

# 测试两种方法
data_direct = test_loads_direct(file_path)
data_readlines = test_loads_readlines(file_path)

# 比较两种方法的结果
if data_direct == data_readlines:
    print("Both methods produce the same result.")
else:
    print("The methods produce different results.")

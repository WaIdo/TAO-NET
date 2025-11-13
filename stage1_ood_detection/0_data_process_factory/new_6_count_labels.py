import json
from collections import Counter

def load_data(file_path):
    """
    从指定文件路径加载 JSON 数据，并返回数据列表。
    """
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

def count_labels(data):
    """
    统计数据中每个标签的出现次数，返回标签计数。
    """
    labels = [item['label'] for item in data]
    label_counts = Counter(labels)
    return label_counts

def main():
    # 训练、验证、测试文件路径
    train_file = './processed_train.json'
    valid_file = './processed_valid.json'
    test_file = './processed_test.json'
    
    # 加载训练、验证、测试数据
    train_data = load_data(train_file)
    valid_data = load_data(valid_file)
    test_data = load_data(test_file)
    
    # 统计每个数据集中的标签数量
    train_counts = count_labels(train_data)
    valid_counts = count_labels(valid_data)
    test_counts = count_labels(test_data)
    
    # 输出训练数据标签统计
    print("Training Data Label Counts:")
    for label, count in train_counts.items():
        print(f"{label}: {count}")
    
    # 分隔符
    print("\n" + "=" * 40 + "\n")
    
    # 输出验证数据标签统计
    print("Validation Data Label Counts:")
    for label, count in valid_counts.items():
        print(f"{label}: {count}")
    
    # 分隔符
    print("\n" + "=" * 40 + "\n")
    
    # 输出测试数据标签统计
    print("Testing Data Label Counts:")
    for label, count in test_counts.items():
        print(f"{label}: {count}")

if __name__ == "__main__":
    main()

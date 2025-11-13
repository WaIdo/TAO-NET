import json
import random
from collections import defaultdict

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    label_count = defaultdict(int)
    for item in data:
        label_count[item['label']] += 1
    
    return data, label_count

def sample_data(data, sample_size):
    label_to_data = defaultdict(list)
    for item in data:
        label_to_data[item['label']].append(item)
    
    sampled_data = []
    sample_count = defaultdict(int)
    for label, items in label_to_data.items():
        sample = random.sample(items, min(sample_size, len(items)))
        sampled_data.extend(sample)
        sample_count[label] = len(sample)
    
    return sampled_data, sample_count

def save_data(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    test_file = '/home/server318/HDD/WHQ/OOD/between-layer-ood-main/0_data_process_factory/processed_test_data.json'
    new_test_file = '/home/server318/HDD/WHQ/OOD/between-layer-ood-main/0_data_process_factory/sampled_test_data.json'
    sample_size = 1134 # 2268 # 1134  # 每个类挑选1134个数据
    
    test_data, original_test_count = load_data(test_file)
    sampled_test_data, sampled_test_count = sample_data(test_data, sample_size)
    
    # 打乱数据
    random.shuffle(sampled_test_data)
    
    save_data(sampled_test_data, new_test_file)
    
    print(f"Sampled and shuffled testing data saved to {new_test_file}")
    
    print("\n原始测试集中各类数据数量：")
    for label, count in original_test_count.items():
        print(f"{label}: {count}")
    
    print("\n抽样后的测试集中各类数据数量：")
    for label, count in sampled_test_count.items():
        print(f"{label}: {count}")

if __name__ == "__main__":
    main()

import json
import random
from collections import defaultdict

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

def sample_data(data, sample_size):
    label_to_data = defaultdict(list)
    for item in data:
        label_to_data[item['label']].append(item)
    
    sampled_data = []
    for label, items in label_to_data.items():
        sampled_data.extend(random.sample(items, min(sample_size, len(items))))
    
    return sampled_data

def save_data(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    train_file = '/home/server318/HDD/WHQ/OOD/between-layer-ood-main/0_data_process_factory/processed_train_data.json'
    new_train_file = '/home/server318/HDD/WHQ/OOD/between-layer-ood-main/0_data_process_factory/sampled_train_data.json'
    sample_size = 10204 # 9070 # 1500 # 10204
    
    train_data = load_data(train_file)
    sampled_train_data = sample_data(train_data, sample_size)
    
    # 打乱数据
    random.shuffle(sampled_train_data)
    
    save_data(sampled_train_data, new_train_file)
    
    print(f"Sampled and shuffled training data saved to {new_train_file}")

if __name__ == "__main__":
    main()

import json
from collections import Counter

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

def count_labels(data):
    labels = [item['label'] for item in data]
    label_counts = Counter(labels)
    return label_counts

def main():
    train_file = '/home/server318/HDD/WHQ/OOD/between-layer-ood-main/0_data_process_factory/processed_train_data.json' # sampled_train_data.json
    test_file = '/home/server318/HDD/WHQ/OOD/between-layer-ood-main/0_data_process_factory/processed_test_data.json'
    
    train_data = load_data(train_file)
    test_data = load_data(test_file)
    
    train_counts = count_labels(train_data)
    test_counts = count_labels(test_data)
    
    print("Training Data Label Counts:")
    for label, count in train_counts.items():
        print(f"{label}: {count}")
    
    
    print("\nTesting Data Label Counts:")
    for label, count in test_counts.items():
        print(f"{label}: {count}")

if __name__ == "__main__":
    main()

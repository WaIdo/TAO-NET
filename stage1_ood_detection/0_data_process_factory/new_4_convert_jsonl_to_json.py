import json
import random

def convert_jsonl_to_json_shuffled(jsonl_file, json_file):
    with open(jsonl_file, 'r') as jsonl_f:
        data = [json.loads(line) for line in jsonl_f]

    random.shuffle(data)

    label_count = {
        "mail": 0,
        "music": 0,
        "youku": 0,
        "taobao": 0,
        "weixin": 0,
        "weibo": 0
    }
# 'mail': 0, 'music': 1, 'youku': 2, 'taobao': 3, 'weixin': 4, 'weibo': 5
    with open(json_file, 'w') as json_f:
        for entry in data:
            label = entry['label']
            label_count[label] += 1
            json.dump(entry, json_f, ensure_ascii=False)
            json_f.write('\n')

    return label_count

# 转换并随机排序训练数据
train_label_count = convert_jsonl_to_json_shuffled('processed_train.jsonl', 'processed_train.json')

# 转换并随机排序验证数据
valid_label_count = convert_jsonl_to_json_shuffled('processed_valid.jsonl', 'processed_valid.json')

# 转换并随机排序测试数据
test_label_count = convert_jsonl_to_json_shuffled('processed_test.jsonl', 'processed_test.json')

print("转换并随机排序完成。结果保存在以下文件中:")
print("- processed_train.json")
print("- processed_valid.json")
print("- processed_test.json")

# 打印统计信息
print("\n训练集中各类数据数量：")
for label, count in train_label_count.items():
    print(f"{label}: {count}")

print("\n验证集中各类数据数量：")
for label, count in valid_label_count.items():
    print(f"{label}: {count}")

print("\n测试集中各类数据数量：")
for label, count in test_label_count.items():
    print(f"{label}: {count}")

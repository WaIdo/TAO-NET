import json
import random

def convert_jsonl_to_json_shuffled(jsonl_file, json_file):
    with open(jsonl_file, 'r') as jsonl_f:
        data = [json.loads(line) for line in jsonl_f]

    random.shuffle(data)

    label_count = {
        "mail": 0,
        "music": 0,
        "taobao": 0,
        "weixin": 0,
        "youku": 0,
        "weibo": 0
    }

    with open(json_file, 'w') as json_f:
        for entry in data:
            label = entry['label']
            label_count[label] += 1
            json.dump(entry, json_f, ensure_ascii=False)
            json_f.write('\n')

    return label_count

# 转换并随机排序训练数据
train_label_count = convert_jsonl_to_json_shuffled('processed_train_data.jsonl', 'processed_train_data.json')

# 转换并随机排序测试数据
test_label_count = convert_jsonl_to_json_shuffled('processed_test_data.jsonl', 'processed_test_data.json')

print("转换并随机排序完成。结果保存在以下文件中:")
print("- processed_train_data.json")
print("- processed_test_data.json")

# 打印统计信息
print("\n训练集中各类数据数量：")
for label, count in train_label_count.items():
    print(f"{label}: {count}")

print("\n测试集中各类数据数量：")
for label, count in test_label_count.items():
    print(f"{label}: {count}")

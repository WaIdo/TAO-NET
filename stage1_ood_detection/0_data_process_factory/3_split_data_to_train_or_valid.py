import json
import random

# 读取原始数据
with open("processed_tinghuaall.jsonl", "r") as file:
    lines = file.readlines()

# 对原始数据进行随机打乱
random.shuffle(lines)

# 将数据按标签分组
data_by_label = {
    "mail": [],
    "music": [],
    "taobao": [],
    "weixin": [],
    "youku": [],
    "weibo": []
}

for line in lines:
    data = json.loads(line)
    label = data["label"]
    data_by_label[label].append(line)

# 划分训练集和测试集
train_data = []
test_data = []

train_count = {
    "mail": 0,
    "music": 0,
    "taobao": 0,
    "weixin": 0
}

test_count = {
    "mail": 0,
    "music": 0,
    "taobao": 0,
    "weixin": 0,
    "youku": 0,
    "weibo": 0
}

for label in ["mail", "music", "taobao", "weixin"]:
    label_data = data_by_label[label]
    train_size = int(0.9 * len(label_data))

    train_data.extend(label_data[:train_size])
    test_data.extend(label_data[train_size:])

    train_count[label] = train_size
    test_count[label] = len(label_data) - train_size

# 将weibo和youku标签的数据全部放入测试集
test_data.extend(data_by_label["youku"])
test_data.extend(data_by_label["weibo"])

test_count["youku"] = len(data_by_label["youku"])
test_count["weibo"] = len(data_by_label["weibo"])


# 写入训练集和测试集数据
with open("processed_train_data.jsonl", "w") as train_file:
    train_file.writelines(train_data)

with open("processed_test_data.jsonl", "w") as test_file:
    test_file.writelines(test_data)

# 打印统计信息
print("数据划分完成。")
print(f"训练集数据保存在 'processed_train_data.jsonl' 文件中,共 {len(train_data)} 行。")
print(f"测试集数据保存在 'processed_test_data.jsonl' 文件中,共 {len(test_data)} 行。")

print("\n训练集中各类数据数量：")
for label, count in train_count.items():
    print(f"{label}: {count}")

print("\n测试集中各类数据数量：")
for label, count in test_count.items():
    print(f"{label}: {count}")

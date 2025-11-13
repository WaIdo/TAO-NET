import json
import os

def convert_label(label):
    label_map = {
        0: "mail",
        1: "music",
        2: "youku",
        3: "taobao",
        4: "weixin",
        5: "weibo"
    }
    return label_map.get(label, None)

def process_file(input_filename, output_filename):
    with open(input_filename, "r") as file, open(output_filename, "w") as output_file:
        for line in file:
            try:
                data = json.loads(line)
                text = data["anchor"]["text"]
                label_dict = data["anchor"]["label"]
                
                # 找到第一个不为 null 的标签值
                label_value = next((v for v in label_dict.values() if v is not None), None)
                
                if label_value is not None:
                    # 将标签值转换为对应的字符串
                    label = convert_label(label_value)
                    if label:
                        # 格式化输出数据
                        output_data = {"text": " ".join(text), "label": label}
                        output_file.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                print(f"警告：在 {input_filename} 中跳过了一个无效的 JSON 条目")

    print(f"转换完成。结果保存在 {output_filename} 文件中。")

# 要处理的文件列表
files_to_process = ["train.txt", "valid.txt", "test.txt"]

for file in files_to_process:
    input_filename = file
    output_filename = f"processed_{os.path.splitext(file)[0]}.jsonl"
    process_file(input_filename, output_filename)

print("所有文件处理完成。")
import json
from collections import defaultdict

def analyze_processed_file(filename):
    print(f"\n分析处理后文件: {filename}")
    
    num_lines_to_print = 0
    label_counts = defaultdict(int)
    
    with open(filename, "r") as file:
        for i in range(num_lines_to_print):
            line = file.readline()
            if not line:
                break
            data = json.loads(line)
            label = data.get("label")
            if label:
                label_counts[label] += 1
            print(f"第 {i + 1} 行:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            print()
        
        for line in file:
            data = json.loads(line)
            label = data.get("label")
            if label:
                label_counts[label] += 1
    
    print("处理后文件中每个类的数据数量:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")
    
    total_count = sum(label_counts.values())
    print(f"文件 '{filename}' 中共有 {total_count} 行。")
    
    return label_counts, total_count

def analyze_original_file(filename):
    print(f"\n分析原始文件: {filename}")
    
    label_counts = defaultdict(int)
    total_count = 0
    
    with open(filename, "r") as file:
        for line in file:
            try:
                data = json.loads(line)
                label_dict = data["anchor"]["label"]
                label_value = next((v for v in label_dict.values() if v is not None), None)
                if label_value is not None:
                    label = convert_label(label_value)
                    if label:
                        label_counts[label] += 1
                total_count += 1
            except json.JSONDecodeError:
                print(f"警告：在 {filename} 中跳过了一个无效的 JSON 条目")
    
    print("原始文件中每个类的数据数量:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")
    
    print(f"文件 '{filename}' 中共有 {total_count} 行。")
    
    return label_counts, total_count

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

# 要分析的文件列表
processed_files = ["processed_train.jsonl", "processed_valid.jsonl", "processed_test.jsonl"]
original_files = ["train.txt", "valid.txt", "test.txt"]

# 分析每个文件并比较结果
for orig_file, proc_file in zip(original_files, processed_files):
    orig_counts, orig_total = analyze_original_file(orig_file)
    proc_counts, proc_total = analyze_processed_file(proc_file)
    
    print(f"\n比较 {orig_file} 和 {proc_file}:")
    if orig_counts == proc_counts and orig_total == proc_total:
        print("数据一致性检查通过：原始文件和处理后文件的类别数量和总行数相同。")
    else:
        print("警告：数据一致性检查失败")
        print("差异:")
        for label in set(orig_counts.keys()) | set(proc_counts.keys()):
            if orig_counts.get(label, 0) != proc_counts.get(label, 0):
                print(f"  {label}: 原始文件 {orig_counts.get(label, 0)}, 处理后文件 {proc_counts.get(label, 0)}")
        if orig_total != proc_total:
            print(f"  总行数: 原始文件 {orig_total}, 处理后文件 {proc_total}")

# 计算所有文件的总统计
print("\n所有文件的总统计:")
total_label_counts_orig = defaultdict(int)
total_label_counts_proc = defaultdict(int)
total_lines_orig = 0
total_lines_proc = 0

for orig_file, proc_file in zip(original_files, processed_files):
    orig_counts, orig_total = analyze_original_file(orig_file)
    proc_counts, proc_total = analyze_processed_file(proc_file)
    
    for label, count in orig_counts.items():
        total_label_counts_orig[label] += count
    total_lines_orig += orig_total
    
    for label, count in proc_counts.items():
        total_label_counts_proc[label] += count
    total_lines_proc += proc_total

print("\n原始文件中每个类的总数据数量:")
for label, count in total_label_counts_orig.items():
    print(f"{label}: {count}")
print(f"所有原始文件共有 {total_lines_orig} 行。")

print("\n处理后文件中每个类的总数据数量:")
for label, count in total_label_counts_proc.items():
    print(f"{label}: {count}")
print(f"所有处理后文件共有 {total_lines_proc} 行。")

if total_label_counts_orig == total_label_counts_proc and total_lines_orig == total_lines_proc:
    print("\n总体数据一致性检查通过：原始文件和处理后文件的总类别数量和总行数相同。")
else:
    print("\n警告：总体数据一致性检查失败")
    print("总体差异:")
    for label in set(total_label_counts_orig.keys()) | set(total_label_counts_proc.keys()):
        if total_label_counts_orig.get(label, 0) != total_label_counts_proc.get(label, 0):
            print(f"  {label}: 原始文件 {total_label_counts_orig.get(label, 0)}, 处理后文件 {total_label_counts_proc.get(label, 0)}")
    if total_lines_orig != total_lines_proc:
        print(f"  总行数: 原始文件 {total_lines_orig}, 处理后文件 {total_lines_proc}")
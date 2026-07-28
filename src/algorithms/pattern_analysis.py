from utils.load_utils import *

def read_action_csv(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            sequence = row[0]  # 假设每行只有一个字符串
            elements = [sequence[i:i + 2] for i in range(0, len(sequence), 2)]  # 每两个字符为一个元素
            sequences.append(elements)
    return sequences

def process_action_file(file_path):
    results = []  # 用于存储每一行的计算结果
    with open(file_path, 'r') as file:
        for line in file:
            items = line.strip().split()
            if len(items) >= 4:  # 确保每行至少有4个元素
                try:
                    result = int(items[2]) + int(items[3])
                    results.append(result)
                except ValueError:
                    print(f"Warning: Line '{line.strip()}' contains non-integer values and will be skipped.")
            else:
                print(f"Warning: Line '{line.strip()}' does not have enough items and will be skipped.")
    return results

def extract_continuous_patterns(sequences, min_support=0.1):
    all_patterns = []
    for sequence in sequences:
        for i in range(len(sequence)):
            for j in range(i + 1, len(sequence) + 1):
                sub_sequence = tuple(sequence[i:j])  # 用元组表示连续子序列
                all_patterns.append(sub_sequence)

    pattern_support = {}
    for pattern in all_patterns:
        if pattern not in pattern_support:
            pattern_support[pattern] = 0
        pattern_support[pattern] += 1

    min_support_count = len(sequences) * min_support
    frequent_patterns = {pattern: support for pattern, support in pattern_support.items() if support >= min_support_count}

    sorted_frequent_patterns = sorted(frequent_patterns.items(), key=lambda x: x[1], reverse=True)
    return sorted_frequent_patterns

def replace_sequences_with_patterns(sequences, patterns, results):
    new_sequences = []
    for sequence, result in zip(sequences, results):
        sequence_str = ''.join(sequence)
        matched_patterns = []
        marked_sequence = sequence_str  # 初始化为原始序列
        pattern_positions = {}  # 记录每个模式的起始位置

        for pattern in sorted(patterns, key=len, reverse=True):  # 按模式长度降序处理
            pattern_str = ''.join(pattern)
            if pattern_str in sequence_str:
                matched_patterns.append(pattern_str)
                marked_sequence = marked_sequence.replace(pattern_str, f"[{pattern_str}]")
                start_index = sequence_str.find(pattern_str)
                position = start_index / len(sequence_str) if len(sequence_str) > 0 else 0
                pattern_positions[pattern_str] = position

        marked_sequence = handle_nested_patterns(marked_sequence)
        new_sequences.append((matched_patterns, marked_sequence, result, pattern_positions))
    return new_sequences

def handle_nested_patterns(marked_sequence):
    stack = []
    new_sequence = []
    for char in marked_sequence:
        if char == '[':
            if stack:
                stack.append(char)
            else:
                new_sequence.append('【')
                stack.append(char)
        elif char == ']':
            stack.pop()
            if not stack:
                new_sequence.append('】')
        else:
            new_sequence.append(char)
    return ''.join(new_sequence)

def get_marked_sequences(log_path, result_path, min_support=0.01):
    # 读取数据
    sequences = read_action_csv(log_path)
    results = process_action_file(result_path)

    # 提取连续子序列的模式
    continuous_patterns = extract_continuous_patterns(sequences, min_support=min_support)

    # 过滤出长度大于等于2的模式
    filtered_continuous_patterns = [(pattern, support) for pattern, support in continuous_patterns if len(pattern) >= 2]

    # 替换 sequences 为序列模式 + 完整的序列列表，并标注模式
    marked_sequences = replace_sequences_with_patterns(sequences, [pattern for pattern, _ in filtered_continuous_patterns], results)

    return marked_sequences
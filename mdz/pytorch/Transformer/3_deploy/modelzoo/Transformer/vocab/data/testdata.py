import numpy as np

# 读取 equal_indices_9487.ftmp 文件中的下标
equal_indices = np.fromfile("equal_indices_9487.ftmp", dtype=np.float32).astype(int)

# 读取 test.txt 文件中的所有行
with open('test.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 根据 equal_indices 中的下标，筛选出 test.txt 中对应的行
filtered_lines = [lines[i] for i in equal_indices if i < len(lines)]

# 将筛选后的行保存到新的文件中
with open('filtered_test.txt', 'w', encoding='utf-8') as file:
    file.writelines(filtered_lines)

print("筛选后的行已保存到 filtered_test.txt 文件中。")
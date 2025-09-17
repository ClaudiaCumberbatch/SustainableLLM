import pandas as pd
import os
from pathlib import Path

# 定义选项标签
choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']

def format_example(df, idx, include_answer=True):
    """格式化单个问题示例"""
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += " {}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += " Answer:"
    if include_answer:
        prompt += " {}".format(df.iloc[idx, k + 1])
    return prompt

def process_csv_files(test_folder='test', output_file='data.csv'):
    """
    处理test文件夹下的所有CSV文件
    
    参数:
    test_folder: 包含CSV文件的文件夹路径
    output_file: 输出文件名
    """
    
    # 确保test文件夹存在
    if not os.path.exists(test_folder):
        print(f"错误：文件夹 '{test_folder}' 不存在")
        return
    
    # 获取所有CSV文件
    csv_files = list(Path(test_folder).glob('*.csv'))
    
    if not csv_files:
        print(f"在 '{test_folder}' 文件夹中没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 存储所有结果
    all_prompts = []
    
    for csv_file in csv_files:
        print(f"处理文件: {csv_file.name}")
        
        try:
            # 读取CSV文件，不使用header
            df = pd.read_csv(csv_file, header=None)
            
            # 为每个问题生成prompt
            for i in range(len(df)):
                # 固定的开头
                prompt = "The following are multiple choice questions about Management Test."
                
                # 添加当前问题（不包含答案）
                prompt += format_example(df, i, include_answer=False)
                
                # 添加到结果列表
                all_prompts.append(prompt)
                
        except Exception as e:
            print(f"处理文件 {csv_file.name} 时出错: {e}")
            continue
    
    # 将结果保存到CSV文件
    if all_prompts:
        # 保存为JSON文件，只有一列prompt
        import json
        with open(output_file.replace('.csv', '.json'), 'w', encoding='utf-8') as f:
            json.dump([{'prompt': p} for p in all_prompts], f, ensure_ascii=False, indent=2)
        print(f"\n成功生成 {len(all_prompts)} 个prompts")
        print(f"结果已保存到 '{output_file.replace('.csv', '.json')}'")
    else:
        print("没有生成任何结果")

# 主程序
if __name__ == "__main__":
    # 可以自定义文件夹路径和输出文件名
    test_folder = 'test'  # CSV文件所在的文件夹
    output_file = 'data2.json'  # 输出文件名
    
    # 处理文件
    process_csv_files(test_folder, output_file)
    
    # 可选：显示输出文件的前几行以验证结果
    # if os.path.exists(output_file):
    #     print(f"\n输出文件 '{output_file}' 的前几行:")
    #     result_df = pd.read_csv(output_file)
    #     print(result_df.head())
    #     print(f"\n总共生成了 {len(result_df)} 行数据")
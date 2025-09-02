import pandas as pd
import numpy as np

def preprocess_data(input_file, output_file):
    """
    预处理数据集：
    1. 筛选role为HN且creation_time不为空的记录
    2. 按creation_time排序
    3. 保存到新文件
    
    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    """
    
    # 读取CSV文件
    print(f"正在读取文件: {input_file}")
    df = pd.read_csv(input_file)
    
    # 显示原始数据信息
    print(f"原始数据行数: {len(df)}")
    print(f"原始数据列: {df.columns.tolist()}")
    
    # 筛选条件：role为HN且creation_time不为空
    # 注意：pandas会将空值读取为NaN
    # mask = (df['role'] == 'HN') & (df['creation_time'].notna())
    mask = (df['creation_time'].notna())
    filtered_df = df[mask].copy()
    
    print(f"筛选后数据行数: {len(filtered_df)}")
    
    # 按creation_time排序（升序）
    filtered_df = filtered_df.sort_values('creation_time', ascending=True)
    
    # 只保留instance_sn和creation_time两列
    filtered_df = filtered_df[['instance_sn', 'creation_time']]
    
    # 将每个时间都减去第一个时间（计算相对时间）
    if len(filtered_df) > 0:
        first_time = filtered_df['creation_time'].iloc[0]
        filtered_df['creation_time'] = filtered_df['creation_time'] - first_time
        print(f"已将所有时间转换为相对时间（相对于第一个时间点: {first_time}）")
    
    # 保存到新文件
    filtered_df.to_csv(output_file, index=False)
    print(f"处理完成！结果已保存到: {output_file}")
    
    # 显示一些统计信息
    print("\n=== 处理结果统计 ===")
    print(f"筛选出的HN记录数: {len(filtered_df)}")
    if len(filtered_df) > 0:
        print(f"最早的creation_time: {filtered_df['creation_time'].min()}")
        print(f"最晚的creation_time: {filtered_df['creation_time'].max()}")
        print(f"\n前5条记录:")
        print(filtered_df.head())
    
    return filtered_df

def main():
    # 设置输入输出文件路径
    input_file = '../dataset/alibaba_2025/disaggregated_DLRM_trace.csv'  # 请修改为您的输入文件路径
    output_file = '../dataset/alibaba_2025/filtered_data.csv'  # 输出文件路径
    
    try:
        # 执行数据预处理
        result_df = preprocess_data(input_file, output_file)
        
        # 可选：显示更多详细信息
        if len(result_df) > 0:
            print("\n=== 数据样例 ===")
            print("前3条记录:")
            print(result_df.head(3).to_string())
            
    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{input_file}'")
        print("请确保文件路径正确")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

# 如果您想直接处理字符串数据（用于测试）
def process_from_string():
    """
    从字符串数据直接处理（用于测试）
    """
    data_string = """instance_sn,role,app_name,cpu_request,cpu_limit,gpu_request,gpu_limit,rdma_request,rdma_limit,memory_request,memory_limit,disk_request,disk_limit,max_instance_per_node,creation_time,scheduled_time,deletion_time
instance_0,HN,app_0,12,12,1,1,25,25,120.0,120.0,640.0,800.0,8,,,
instance_1,HN,app_0,12,12,1,1,25,25,120.0,120.0,680.0,800.0,8,,,
instance_2,HN,app_0,12,12,1,1,25,25,120.0,120.0,680.0,800.0,8,,,
instance_12726,CN,app_87,64,64,0,0,100,100,320.0,320.0,255.0,300.0,2,1241953.0,1241953.0,1247275.0
instance_12727,CN,app_126,96,96,0,0,1,1,480.0,480.0,425.0,500.0,-1,1242286.0,1242286.0,
instance_12728,HN,app_20,8,8,1,1,25,25,40.0,40.0,85.0,100.0,2,1242636.0,1242636.0,1242641.0"""
    
    from io import StringIO
    
    # 将字符串转换为DataFrame
    df = pd.read_csv(StringIO(data_string))
    
    # 应用筛选条件
    mask = (df['role'] == 'HN') & (df['creation_time'].notna())
    filtered_df = df[mask].copy()
    
    # 按creation_time排序
    filtered_df = filtered_df.sort_values('creation_time', ascending=True)
    
    # 只保留instance_sn和creation_time两列
    filtered_df = filtered_df[['instance_sn', 'creation_time']]
    
    # 将每个时间都减去第一个时间（计算相对时间）
    if len(filtered_df) > 0:
        first_time = filtered_df['creation_time'].iloc[0]
        filtered_df['creation_time'] = filtered_df['creation_time'] - first_time
        print(f"\n原始第一个时间点: {first_time}")
        print("已转换为相对时间")
    
    print("\n筛选并排序后的结果:")
    print(filtered_df.to_string())
    
    # 保存结果
    filtered_df.to_csv('test_output.csv', index=False)
    print("\n结果已保存到 test_output.csv")
    
    return filtered_df

if __name__ == "__main__":
    # 选择运行模式
    # 1. 处理实际文件
    main()
    
    # 2. 或者使用测试数据（取消下面的注释）
    # process_from_string()
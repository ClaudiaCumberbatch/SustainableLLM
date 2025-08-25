import requests
import csv
import json
from datetime import datetime
import os
import subprocess
import time
import signal

headers = []

def send_request(url, data, csv_filename):
    global headers
    # 发送请求并获取响应
    with requests.post(url, json=data, stream=True) as response:

        if response.status_code == 200:
            response_data = response.json()
        
        # 写入CSV文件（单行格式）
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            file_is_empty = os.stat(csv_filename).st_size == 0
            if file_is_empty:
                headers = list(response_data.keys())
                writer.writerow(headers)

            # 写入数据行
            row = []
            for key in headers:
                value = response_data[key]
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                row.append(value)
            writer.writerow(row)


if __name__ == "__main__":

    url = "http://localhost:11434/api/generate"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"response_{timestamp}.csv"
    power_log_filename = f"gpu_power_{timestamp}.log"

    # 启动power monitor进程
    print("Starting power monitor...")
    power_monitor_process = subprocess.Popen([
        "nvidia-smi", 
        "--query-gpu=timestamp,index,power.draw", 
        "--format=csv", 
        "-lms", "10"
    ], stdout=open(power_log_filename, 'w'), stderr=subprocess.PIPE)
    
    # 给监控进程一点时间启动
    time.sleep(3)
    
    try:
        # 记录推理开始时间
        inference_start_time = time.time()
        
        for i in range(100):
            data = {
                "model": "llama3.2",
                "prompt": "Why is the sky blue?",
                "stream": False,
                "options": {
                    "num_predict": i*20
                }
            }
            print(f"Sending request {i}/100 with num_predict={i*20}...")
            send_request(url, data, csv_filename)
        
        # 记录推理结束时间
        inference_end_time = time.time()
        inference_duration = inference_end_time - inference_start_time
        print(f"Inference completed in {inference_duration:.2f} seconds")
        
    finally:
        # 确保无论发生什么错误都会停止监控进程
        print("Stopping power monitor...")
        
        # 尝试优雅地终止进程
        power_monitor_process.terminate()
        
        # 等待一段时间让进程正常退出
        try:
            power_monitor_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # 如果进程没有正常退出，强制杀死
            print("Power monitor process did not terminate gracefully, forcing kill...")
            power_monitor_process.kill()
            power_monitor_process.wait()
        
        print(f"Power monitoring data saved to {power_log_filename}")
        print(f"Response data saved to {csv_filename}")
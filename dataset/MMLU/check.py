import chardet

def check_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read(10000)  # 只读取前10000字节即可
        result = chardet.detect(rawdata)
        print(f"文件 {file_path} 的编码格式为: {result['encoding']}，置信度: {result['confidence']}")

# check_encoding('data.json')
check_encoding('../alpaca/alpaca_data.json')
# 打开txt文件进行读取
with open('/mnt/petrelfs/zhanjun.p/mllm/mmgpt/src/m_utils/instructions_txt/text/text2audio.txt', 'r', encoding='utf-8') as txt_file:
    lines = txt_file.readlines()

# 去除每行的换行符
lines = [line.strip() for line in lines]

# 创建一个包含字符串的列表
string_list = lines

# 写入.py文件
with open('instructions.py', 'w', encoding='utf-8') as py_file:
    py_file.write('text2other_instructions = {\n')
    py_file.write('    "audio": [\n')
    
    for string in string_list:
        string=string.split(".")[1].strip()
        if string[-1] != '.' and string[-1] != '?' and string[-1] != '!':
            string += '.'
        print(string)
        if len(string) > 1:
            py_file.write('        "{}",\n'.format(string))
    
    py_file.write('    ]\n')
    py_file.write('}\n')

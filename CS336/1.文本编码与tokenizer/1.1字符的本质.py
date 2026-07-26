char_en='A'
print(f'字符：{char_en},内存(ASCII/Unicode): {ord(char_en)}')
#ord把字符转换为Unicode码点(十进制)
char_cn='中'
print(f'字符：{char_cn},Unicode码点(十进制): {ord(char_cn)}')
print(f'字符：{char_cn},Unicode码点(十六进制): {hex(ord(char_cn))}')


text='中'
#编码：字符->字节
encoded_bytes=text.encode('utf-8')
print(f'原始文本:{text}')
print(f'UTF-8编码后的字节序列:{encoded_bytes}')
print(f'字节列表:{[i for i in encoded_bytes]}')
#'中'—> UTF-8编码后是3个字节，分别是0xe4,0xb8,0xad

print(ord('中'))  # 输出Unicode码点(十进制)
print(hex(ord('中')))  # 输出Unicode码点(十六进制)
print(bin(ord('中')))  # 输出Unicode码点(二进制)
print('牛'.encode('utf-8'))  # 输出UTF-8编码后的字节序列
#根据字节表和二进制，从后往前填充得到字节序列
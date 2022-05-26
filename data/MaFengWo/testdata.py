import numpy as np
import pandas as pd
import os

path = '/home/admin123/ruxia/HAN-CDGRccnu/Experiments/MaFengWo/NATRbce/data/'

file1 = open(path + 'userRatingNegative.txt', 'r', encoding='utf-8') # 打开要去掉空行的文件
file2 = open(path + 'newuserRatingNegative.txt', 'w', encoding='utf-8') # 生成没有空行的文件

for line in file1.readlines():    
    if line == '\n':        
       line = line.strip("\n")       
    file2.write(line)

print('输出成功....')  
file1.close()
file2.close()

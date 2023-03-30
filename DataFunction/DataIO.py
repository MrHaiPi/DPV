import csv
import os
import shutil
import numpy as np

def DelPath(path):
    shutil.rmtree(path)

def DelFile(path):
    if os.path.exists(path) == False:
        return
    for i in os.listdir(path):
       path_file = os.path.join(path, i)
       if os.path.isfile(path_file):
         os.remove(path_file)
       else:
           DelFile(path_file)

def WriteCSV(data, headers, fileName, isAppend=False):
    if isAppend:
        with open(fileName, "a+", newline='')as f:
            f_csv = csv.writer(f)
            f_csv.writerows(data)
    else:
        with open(fileName, 'w', newline='')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(data)

def ReadCSV(fileName):
    with open(fileName)as f:
        f_csv = csv.reader(f)
        rows = []
        for row in f_csv:
            rows.append(row)
        return rows

def GetFileName(fileDir):
  for root, dirs, files in os.walk(fileDir):
      return files
      #print(root) #当前目录路径
      #print(receDirs) #当前路径下所有子目录
      #print(files) #当前路径下所有非目录子文件

def GetDirName(fileDir):
  for root, dirs, files in os.walk(fileDir):
      return dirs
      #print(root) #当前目录路径
      #print(receDirs) #当前路径下所有子目录
      #print(files) #当前路径下所有非目录子文件

def CopyFile(srcfile, dstpath):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        # 分离文件名和路径
        fpath, fname = os.path.split(srcfile)
        if not os.path.exists(dstpath):
            # 创建路径
            os.makedirs(dstpath)
        # 复制文件
        shutil.copy(srcfile, dstpath + '/' + fname)

def MakeDir(path):
    if not os.path.exists(path):
               os.makedirs(path)

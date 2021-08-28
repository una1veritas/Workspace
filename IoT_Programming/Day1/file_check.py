#coding:utf-8
import os.path
import sys

filename = sys.argv[1]
print('Check the file/path '+filename)

if os.path.exists(sys.argv[1]):
    print(u"ありまぁす！.")
else:
    print(u"ありません．")

linenumber = 0
with open(filename,'r') as f:
    for row in f:
        print(linenumber, row.strip())
        linenumber += 1
f.close()

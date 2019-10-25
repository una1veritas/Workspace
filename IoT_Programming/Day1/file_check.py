#coding:utf-8
import os.path
import sys

print('Checks the file/path '+sys.argv[1])

if os.path.exists(sys.argv[1]):
    print(u"ありまぁす！.")
else:
    print(u"ありません．")

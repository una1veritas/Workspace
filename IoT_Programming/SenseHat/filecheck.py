#coding:utf-8
import os.path
import sys

if os.path.exists(sys.argv[1]):
    print(u"ありまぁす！.")
else:
    print(u"なかった．")

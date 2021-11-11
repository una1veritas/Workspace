import os
import sys

fullpath_name = u'/Users/sin/Dropbox/アルゴリズム設計Ａ/202104/data'

# p = pathlib.Path(fullpath_name)
# print(p.stat())

print(fullpath_name)
if os.path.isdir(fullpath_name) :
    for filename in sorted(os.listdir(fullpath_name)):
        if os.path.isdir(os.path.join(fullpath_name,filename)) and filename.startswith('C11L202-2021-0101-1121010101-'):
            print(filename)
# print(os.path.getmtime(fullpath_name))
# # 1549094615.9723485
#
# print(os.path.getctime(fullpath_name))
# # 1549094615.9723485
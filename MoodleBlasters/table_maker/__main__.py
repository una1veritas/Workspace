import os
import pathlib
import datetime
import time
import platform

fullpath_name = u'/Users/sin/Dropbox/離散数学Ⅱ-出席＋レポート'

p = pathlib.Path(fullpath_name)
print(p.stat())

print(os.path.getmtime(fullpath_name))
# 1549094615.9723485

print(os.path.getctime(fullpath_name))
# 1549094615.9723485
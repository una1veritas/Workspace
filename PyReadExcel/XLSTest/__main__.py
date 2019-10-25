import os
import sys

args = sys.argv[1:]
print(args)
path_script = os.path.dirname(os.path.normpath(os.path.expanduser(args[0]))) 
print(path_script)
path_script = os.path.dirname(os.path.abspath(__name__)) 
print(path_script)
#絶対パスと相対パスをくっつける
path_joined = os.path.join(path_script, '../')

#正規化して絶対パスにする
path_data = os.path.normpath(path_joined)

print(path_data)

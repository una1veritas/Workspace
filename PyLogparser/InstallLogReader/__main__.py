#
#
import sys

params = dict()
if sys.argv[1][0] == '-' :
    [attr, val] = sys.argv[1][1:].split('=')
    params[attr] = val

print(params)
filename = params['file']

file = open(filename, 'r', encoding='utf-8')
cont = file.read()
file.close()

print(cont)
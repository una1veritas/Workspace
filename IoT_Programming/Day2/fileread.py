# -*- coding: utf-8 -*-
i = 0
with open('test.py','r') as f:
    for row in f:
        print(i, row.strip())
        i += 1

print("file io read finished.")

f = open('writetest.txt','w')
for i in range(20):
    f.write('write test ')
    f.write(str(i))
    f.write('\n')

f.close()

print("file io write finished.")

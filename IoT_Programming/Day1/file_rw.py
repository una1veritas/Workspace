# -*- coding: utf-8 -*-
import sys

print('Read text lines from file.')
i = 0
with open(sys.argv[1],'r') as f:
    for row in f:
        print(i, row.strip())
        i += 1

print("Write text to file \'test_out.txt\'.")

f = open('test_out.txt','w')
for i in range(20):
    f.write('write test ')
    f.write(str(i))
    f.write('\n')

f.close()

print("Finished.")

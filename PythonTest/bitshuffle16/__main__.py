'''
Created on 2026/01/04

@author: sin
'''
import random

'''[11, 1, 14, 9, 13, 10, 8, 3, 0, 12, 15, 7, 2, 6, 5, 4]'''

def shuffle16(val16, shuffled): 
    val16 ^= 0xa55a
    res16 = 0
    for i in shuffled :
        res16 <<= 1
        res16 |= (val16>>shuffled[i]) & 1
    return res16

if __name__ == '__main__':
    remained = set(range(0x10000))
    shuffle_seq = list(range(16))
    random.shuffle(shuffle_seq)
    for i in range(0, 0x10000) :
        remained.remove(shuffle16(i, shuffle_seq))
        print(f'{shuffle16(i, shuffle_seq):04x}')
    print(remained)
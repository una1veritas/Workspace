'''
Created on 2026/01/04

@author: sin
'''
import random

def permutation_random(seq : list, seed: int | None = None) -> list[int]:
    rng = random.Random(seed)  # pass None for non-deterministic
    rng.shuffle(seq)
    return seq

def murmur3a_hash(x) :
    x ^= x>>16
    x *= 0x85ebca6b
    x ^= x >> 13
    x *= 0xc2b2ae35
    x ^= x >> 16
    return x & 0xffffffff

if __name__ == '__main__':
    rndtbl = permutation_random(list(range(256)))
    for i in range(256):
        if i % 16 == 0 :
            print()
        print(f'0x{rndtbl[i]:02x}, ', end='')
        
    
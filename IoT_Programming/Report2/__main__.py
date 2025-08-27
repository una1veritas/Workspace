'''
Created on 2025/08/26

@author: sin
'''
import math, sys

if __name__ == '__main__':
    nth_limit = 1000
if len(sys.argv) > 1 :
    try:
        nth_limit = int(sys.argv[1])
    except ValueError:
        print(f'not interpreted as number: {sys.argv[1]}')
    except:
        print('something going wrong.')
    print(f'trying to enumerate up to about {nth_limit}th prime.', end='\n\n')
    sievesize = 200 if nth_limit < 50 else int(nth_limit * (math.ceil(1.2 * math.log(nth_limit, math.e))))

    is_prime = [True] * sievesize
    is_prime[0] = False   # primes[0] は使わない
    is_prime[1] = False   # 1 は素数でない

    for i in range(2, sievesize) :
        if is_prime[i] :
            for comp_num in range(i*2, sievesize, i):  # i の整数倍について
                is_prime[comp_num] = False
    
    count = 0
    for i in range(sievesize):
        if is_prime[i] or count == 0 :
            if count % 100 == 0 :
                print('        +---------------------------------------------------------------------')
                print('        ', end = '')
                for j in range(10):
                    print(f'{j:5}  ', end='')
                print('\n        +---------------------------------------------------------------------')

            if count % 10 == 0 :
                print(f"{count:5}th |", end='')
                if not is_prime[i] :
                    print("        ", end="")
                else:
                    print(f" {i:5}, ", end="")
            elif count % 10 < 9 :
                print(f"{i:5}, ", end = '')
            else:
                print(f"{i:5}")
            count += 1

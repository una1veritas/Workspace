'''
Created on 2018/03/25

@author: sin
'''
import sys
import fibonacci

def main(argv):
    for arg in argv[1:]:
        argval = int(arg)
        print(fibonacci.series(argval), end=', ') #, fibonacci.series(argval+1)/fibonacci.series(argval))
    print()
    return

if __name__ == '__main__':
    main(sys.argv)
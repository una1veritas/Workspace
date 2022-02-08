'''
Created on 2018/03/25

@author: sin
'''
import sys
import fibonacci

def main(argv):
    args = list()
    for arg in argv[1:]:
        args.append(int(arg))
    if len(args) == 2 :
        print(fibonacci.series(args[0], args[1]))
    else: 
        print(fibonacci.series(args[0]))
    print()
    return

if __name__ == '__main__':
    main(sys.argv)
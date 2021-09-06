import sys
import math
import time

def collatz_next(n):
    if type(n) is not int :
        return n
    if (n & 1) == 0 :
        return n>>1
    else:
        return (n<<1) + n + 1

if __name__ == "__main__" :    
    if not ( len(sys.argv) > 1 ) :
        print("requires an integer.\n", file=sys.stderr)
        exit()
    
    try:
        swatch = time.time()
        histgram = dict()
        stopn = int(eval(sys.argv[1]))
        n = 1
        while (stopn < 0) or (n < stopn) :
            if 1 << (int(math.log2(n))) == n or n + 1 == stopn:
                prn = True
            else:
                prn = False
            i = n
            count = 0
            if prn: print(i, end=" ")
            while 1 << (int(math.log2(i))) != i :
                i = collatz_next(i)
                count += 1
            if i not in histgram :
                histgram[i] = 1
            else:
                histgram[i] += 1
            if prn:
                if count :
                    print("("+str(count)+") "+str(i), end= "")
                print(", passed "+str(round(time.time() - swatch, 3))+ " sec.")
            n += 1
    except KeyboardInterrupt:
        print("Quit by ctrl-C.")
        pass
    print(sorted(histgram.items(), key=lambda x:x[1], reverse=True))
    
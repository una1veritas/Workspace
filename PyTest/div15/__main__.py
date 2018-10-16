import sys

def div15(val):
    quot = (val*17) >> 8
    if not (val - 15 - quot * 15) :
        quot = quot + 1
    return quot

limval = eval(sys.argv[1])

for v in range(0,limval):
    if int(v/15) != div15(v):
        print(str(v)+', '+str(int(v/15))+', '+str(div15(v)))

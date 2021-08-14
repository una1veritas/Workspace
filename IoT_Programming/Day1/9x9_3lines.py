for p in [(i, j) for i in range(1,10) for j in range(1,10)]:
    print("{0:4}".format(p[0]*p[1]), end="")
    if p[1] == 9 : print()


mydict = {}

def do_command(tlist):
    cmd = tlist[0]
    if cmd == "add" :
        if len(tlist) > 2 :
            mydict[tlist[1]] = tlist[2]
        else:
            mydict[tlist[1]] = ""
    elif cmd == "test" :
        if tlist[1] in mydict :
            print "Has key ",tlist[1],", value ", mydict[tlist[1]]
        else:
            print "Not found key ", tlist[1]
    elif cmd == "remove" :
        mydict.pop(tlist[1])
    elif cmd == "clear" :
        mydict.clear()
    else:
        print "??? ", tlist[0]
    print mydict

while True:
    strvar = raw_input("Enter something: ")
    print "You've entered ", strvar, ".\n"
    if len(strvar) == 0 :
        continue
    tokens = strvar.split()
    if tokens[0] == "quit" :
        break
    do_command(tokens)

print "Ok, bye."
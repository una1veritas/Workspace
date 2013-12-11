import sys

lnum = 0

while True:
    strline = sys.stdin.readline().strip()
    if strline == "END" :
        print "Encountered the end."
        break
    if not strline :
        break
    print "line ", lnum, ": ", strline
    number, strline = strline.split(" ", 1)
    aord, hintstr = strline.split(" ", 1)
    print "No. ", number, ", Across/Down ", aord, ": ", hintstr
    lnum = lnum + 1
#end of while

print "Finished."

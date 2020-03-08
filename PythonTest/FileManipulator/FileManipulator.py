import sys
from pathlib import Path
import re

p = Path(sys.argv[1])

for entry in list(p.glob('**/*')) :
    if entry.name[:1] == '.' :
        continue
    fullpathname = str(entry)
    if '(' in entry.name :
        print(fullpathname)
    #    fullpathname = re.sub('[ ]*\([^\)]*\)[ ]*', '', fullpathname)
    #fullpathname = fullpathname.encode('cp932', "ignore") #.decode('CP932')
    #if fullpathname != str(entry) :
        #entry.rename(fullpathname)
        #print(str(entry), fullpathname)
        #break
    #elif '"' in str(entry):
    #    print(str(entry))
    elif entry.name[-1] == '.' :
        print(str(entry))
    elif "/" in entry.name or ':' in entry.name:
        print(str(entry))
        
    
print('bye.')
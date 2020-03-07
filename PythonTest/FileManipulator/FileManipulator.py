import sys
from pathlib import Path
import re

p = Path(sys.argv[1])

while True:
    for entry in list(p.glob('**/*')) :
        if entry.name[:1] == '.' :
            continue
        fullpathname = str(entry)
        if '(' in fullpathname :
            fullpathname = re.sub('[ ]*\([^\)]*\)[ ]*', '', fullpathname)
        fullpathname = fullpathname.encode('cp932', "ignore").decode('CP932')
        if fullpathname != str(entry) :
            entry.rename(fullpathname)
            print(fullpathname)
            break
    else:
        break
    
print('bye.')
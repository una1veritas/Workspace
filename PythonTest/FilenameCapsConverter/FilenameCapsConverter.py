import sys
from pathlib import Path
import re

p = Path(sys.argv[1])

for entry in list(p.glob('**/*')) :
    if entry.name[:1] == '.' :
        continue
    #filteredname = str(entry).encode('cp932', "ignore").decode('CP932')
    if '(' in str(entry):
        fullpath = str(entry)
        fullpath = re.sub('[ ]*\([^\)]*\)[ ]*', '', fullpath)
        entry.rename(fullpath)
        print(fullpath)
    
    
print('bye.')
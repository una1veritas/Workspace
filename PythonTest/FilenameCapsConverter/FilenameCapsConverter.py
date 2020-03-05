import sys
from pathlib import Path
import re

p = Path(sys.argv[1])

for entry in list(p.glob('**/*')) :
    if entry.suffix == '.MID' :
        print(str(entry))
        lentry = re.sub('.MID$', '.mid', str(entry))
        entry.rename(lentry)
    
    
print('bye.')
import sys

file_name = ''
load_addr = 0
argix = 1
while argix < len(sys.argv):
    if sys.argv[argix] == '-l' :
        argix += 1
        load_addr = int(sys.argv[argix], 16) 
        argix += 1
    elif len(file_name) == 0 :
        file_name = sys.argv[argix]
        argix += 1
    else :
        print('Usage: bin2array.py [-l load_addr] file_name')
        sys.exit(1)

try:
    f = open(file_name, 'rb') 
except FileNotFoundError:
    print('Error: cannot open file '.file_name)
else:
    with f:
        count = 0
        while (b := f.read(1)):
            #print(type(b))
            print('0x{:02x}, '.format(int.from_bytes(b, "big")), end='')
            count += 1
            if count % 16 == 0 :
                print() 
            if count % 256 == 0 :
                print(f'// {(load_addr+count):04x}')
print()

import sys

def hashCode(s):
    total = 0
    for index in range(0, min(len(s), 3)):
        total = total*31 + ord(s[index])
    return total

def add_to_hashtable(s, table):
    table[ hashCode(s) % len(table) ].append(s)

def remove_from_hashtable(s, table):
    if s in table[ hashCode(s) % len(table) ] :
        table[ hashCode(s) % len(table) ].remove(s)
    
if len(sys.argv) == 1 :
    exit
    
args = sys.argv[1:]

table_size = 7
hashtable = {}
for ix in range(0, table_size):
    hashtable[ix] = [ ]  # i,...,table_size-1 の各添え字にカラのリストを対応させる

print('add inputs: '+str(args))
for s in args:
    add_to_hashtable(s, hashtable)    
for keyindex in hashtable:
    print(keyindex, hashtable[keyindex])
print()

print('add \'olive\' and \'fortune\'')
add_to_hashtable('olive', hashtable)
add_to_hashtable('fortune', hashtable)
for keyindex in hashtable:
    print(keyindex, hashtable[keyindex])
print()

print('remove \'olive\'')
remove_from_hashtable('olive', hashtable)
for keyindex in hashtable:
    print(keyindex, hashtable[keyindex])
print()

'''
Betelgeuse:PyHashTable sin$ python3 TableTest/main.py awake arise or be forever fallen
add inputs: ['awake', 'arise', 'or', 'be', 'forever', 'fallen']
0 ['forever']
1 ['fallen']
2 []
3 ['be']
4 ['awake', 'arise']
5 []
6 ['or']

add 'olive' and 'fortune'
0 ['forever', 'olive', 'fortune']
1 ['fallen']
2 []
3 ['be']
4 ['awake', 'arise']
5 []
6 ['or']

remove 'olive'
0 ['forever', 'fortune']
1 ['fallen']
2 []
3 ['be']
4 ['awake', 'arise']
5 []
6 ['or']
'''
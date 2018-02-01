import sys

def hashCode(s):
    total = 0
    for index in range(0, min(len(s), 3)):
        total = total*31 + ord(s[index])
    return total

def init_hashtable(table, size_n):
    table.clear()
    for i in range(0, size_n) :
        table.append( [ ] )
    return table

def in_hashtable(s, table) :
    return s in table[ hashCode(s) % len(table) ]

def add_to_hashtable(s, table):
    if not in_hashtable(s,table):
        table[ hashCode(s) % len(table) ].append(s)

def remove_from_hashtable(s, table):
    if in_hashtable(s,table) :
        table[ hashCode(s) % len(table) ].remove(s)
    
if len(sys.argv) == 1 :
    exit
    
args = sys.argv[1:]

hashtable = [ ]
init_hashtable(hashtable, 7)

print('add inputs: '+str(args))
for s in args:
    add_to_hashtable(s, hashtable)
for i in range(0, len(hashtable)):
    print(i, hashtable[i])
print()

print('add \'olive\' and \'fortune\'')
add_to_hashtable('olive', hashtable)
add_to_hashtable('fortune', hashtable)
for i in range(0, len(hashtable)):
    print(i, hashtable[i])
print()

print('remove \'olive\'')
remove_from_hashtable('olive', hashtable)
for i in range(0, len(hashtable)):
    print(i, hashtable[i])
print()

'''
$ python3 TableTest/main.py awake arise or be forever fallen
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
import sys

path = 'test.tm'

states = set()
alphabet = set()
trans = [ ]
try:
    # ファイルオブジェクトを変数fに代入。close()は書かなくてＯＫ。
    with open(path, mode='r', encoding='utf-8') as tfile:
        for a_line in tfile:
            trans.append(a_line.split())
except:
    print(path + 'の読み込みに失敗しました。')

# テープ数を推測する
num = len(trans[0])
if ( num % 3 == 0 ):
    num_tape = num // 3
    read_only = False
else:
    num_tape = (num + 1) // 3
    read_only = True
    
print('num_tape = ', num_tape)
for a_tuple in trans:
    s = a_tuple[0]
    if s[0] == '!' :
        s = s[1:]
        a_tuple[0] = s
    states.add(s)
    s = a_tuple[1+num_tape]
    if s[0] == '!' :
        s = s[1:]
        a_tuple[1+num_tape] = s
    states.add(s)
    
    for i in range(1,num_tape) :
        alphabet.add(a_tuple[i])
    if not read_only :
        for i in range(1+num_tape+1,1+num_tape+1+num_tape*2,2):
            alphabet.add(a_tuple[i])
    else:
        for i in range(1+num_tape+1+1,1+num_tape+1+1+(num_tape-1)*2,2):
            alphabet.add(a_tuple[i])

print(alphabet)        
print(states)
print(trans)
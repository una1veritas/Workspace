
import itertools

def color(i):
    if 1 <= i <= 3 :
        return 'b'
    elif 4<= i <= 7 :
        return 'r'
    elif 7 <= i <= 12 :
        return 'w'
    return 'N.A.'
    
def ball_seq_str(perm):
    t = '['
    for no in perm :
        t += color(no) + ' '
    t += ']'
    return t

def color_touching(seq, col):
    for i in range(0,len(seq)-1) :
        if color(seq[i]) == col and color(seq[i]) == color(seq[i+1]) :
            return True
    return False

l = [i for i in range(1,13)]

counter = 0
for perm in itertools.permutations(l, 12):
    if not color_touching(perm,'r') :
        print(ball_seq_str(perm))
        counter += 1

print('finished.', counter)
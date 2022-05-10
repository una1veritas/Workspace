'''
Created on 2022/04/13

@author: sin
'''
from numba.core.types import none

class Hai:
    _TSU = '🀀🀁🀂🀃🀆🀅🀄🀫'
    _MAN = '🀇🀈🀉🀊🀋🀌🀍🀎🀏'
    _BUM = '🀐🀑🀒🀓🀔🀕🀖🀗🀘'
    _PIN = '🀙🀚🀛🀜🀝🀞🀟🀠🀡'
    
    def __init__(self, val):
        if isinstance(val, str):
            if len(val) == 2 :
                k = {'t': 0, 'm': 1, 's': 2, 'p': 3}[val[0].lower()]
                if val[1] in '123456789' :
                    v = int(val[1]) - 1
                elif val[1] in 'eswnpfc':
                    v = {'e': 0, 's': 1, 'w': 2, 'n': 3, 'p': 4, 'f': 5, 'c': 6}[val[1]]
                else:
                    raise ValueError('Invalid value '+str(val))
                red = 3 if val[0].isupper() else 0
                self.id = (k<<6) | (v<<2) | (red)
        elif isinstance(val, int):
            k = (val>>6) & 0x3
            v = (val>>2) & 0xf
            if k == 0 :
                if v > 6 :
                    raise ValueError('Invalid number '+str((k, v)))
            else:
                if v > 8 :
                    raise ValueError('Invalid number '+str((k, v)))
            r = val & 0x3                    
            self.id = (k<<6) | (v<<2) | r
        return
    
    def kind(self):
        return "tmsp"[(self.id >> 6) & 0x3]
    
    def number(self):
        if self.kind() == 't' :
            return 0
        return ((self.id>>2) & 0xf) + 1
    
    def __str__(self):
        if self.kind() == 't':
            return self._TSU[(self.id >> 2) & 0xf]
        elif self.kind() == 'm':
            return self._MAN[(self.id >> 2) & 0xf]
        elif self.kind() == 's':
            return self._BUM[(self.id >> 2) & 0xf]
        elif self.kind() == 'm':
            return self._PIN[(self.id >> 2) & 0xf]
        
if __name__ == '__main__':
    for i in range(164):
        try:
            print(Hai(i),end='')
        except ValueError:
            pass #print(i, end = ' ')
    print()
'''
Created on 2022/04/13

@author: sin
'''
import Crypto.Random.random
import random
import string

class MahjongTile:
    SUITS = ('man', 'pin', 'sou', 'wind', 'dragon')
    SUITS_KANJI = (u'萬', u'筒', u'索', u'風', u'龍')
    RANK_KANJI = u'一二三四伍六七八九'
    WINDS_RANK = ('east', 'south', 'west', 'north')
    DRAGONS_RANK = ('white', 'green', 'red')
    WINDS_RANK_KANJI = (u'東', u'南', u'西', u'北')
    DRAGONS_RANK_KANJI = (u'　', u'發', u'中')
    UTF_CHAR = { 'wind': '🀀🀁🀂🀃', 'dragon' : '🀆🀅🀄🀫', 'man': '🀇🀈🀉🀊🀋🀌🀍🀎🀏', 'sou': '🀐🀑🀒🀓🀔🀕🀖🀗🀘', 'pin': '🀙🀚🀛🀜🀝🀞🀟🀠🀡' }
    
    def __init__(self, name = '', suit=None, rank=None, red=False):
        """
        Initialize a Mahjong tile.
        :param type: The suit of the tile ('man', 'pin', 'sou', 'honor').
        :param rank: The rank of the tile (1-9),or the order number of honor tile ('east', 'south', 'west', 'north', 'white', 'green', 'red').
        """
        if len(name) > 0 :
            if name[0] in '123456789' and name[1] in 'mps' :
                self.rank = int(name[0])
                self.suit = {'m':'man', 'p':'pin', 's':'sou'}[name[1]]
                self.red = red
                return
            elif name[0].upper() in ('E', 'S', 'W', 'N') :
                self.suit = 'wind'
                self.rank = name[0].upper()
                self.red = False
                return
            elif name[0].upper() in ('W', 'G', 'R') :
                self.suit = 'dragon'
                self.rank = name[0].upper()
                self.red = False
                return 
            else:
                raise ValueError(f"Invalid name: {name}.")
                    
        if suit not in self.SUITS:
            raise ValueError(f"Invalid sutype: {suit}. Valid suits are {self.SUITS}.")
        if suit in ('pin', 'man', 'sou') and 1 <= rank <= 9 :
            self.suit = suit
            self.rank = rank
            self.red = False
            return 
        elif suit in ('wind', 'dragon') and rank.lower() in self.WINDS_RANK + self.DRAGONS_RANK:
            self.suit = suit
            self.rank = rank.lower()
            self.red = False
            return
        raise ValueError(f"Invalid suit or rank: {suit}, {rank}.")
    
    def __repr__(self):
        return f"MahjongTile(suit='{self.suit}', rank='{self.rank}')"
    
    def __str__(self):
        if self.is_suited() :
            #return f'[{self.RANK_KANJI[self.rank - 1]+self.SUITS_KANJI[self.SUITS.index(self.suit)]}]'
            if self.red :
                return f'({str(self.rank)+self.suit[0]})'                
            return f'[{str(self.rank)+self.suit[0]}]'
        return f'[{self.rank[0].upper()} ]'
        
    def __eq__(self, other):
        return self.suit == other.suit and self.rank == other.rank
    
    def __lt__(self, other):
        if not isinstance(other, MahjongTile):
            return NotImplemented
        if self.suit == other.suit :
            if self.suit in ('man', 'sou', 'pin') :
                return self.rank < other.rank
            return (self.WINDS_RANK + self.DRAGONS_RANK).index(self.rank) < (self.WINDS_RANK + self.DRAGONS_RANK).index(other.rank) 
        return self.SUITS.index(self.suit) < self.SUITS.index(other.suit)
    
    def is_honor(self):
        return self.suit in ('wind', 'dragon')

    def is_wind(self):
        return self.suit == 'wind'

    def is_dragon(self):
        return self.suit == 'dragon'

    def is_suited(self):
        return self.suit in ('man', 'pin', 'sou')
    
    def dora_next(self):
        return MahjongTile(suit=self.suit, rank = self.rank + 1)
    
class MahjongPlayer:
    def __init__(self, name = None):
        if name == None :
            chars = string.ascii_letters + string.digits
            random_string = ''.join(random.choice(chars) for _ in range(8))
            name = random_string
        self.player_name = name
        self.hand = list()
        self.opened = list()
    
    def name(self):
        return self.player_name
    
    def clear(self):
        self.hand.clear()
        self.opened.clear()
    
class MahjongTable: 
    def __init__(self):
        self.tile_set = list()
        for a_suit in MahjongTile.SUITS:
            if a_suit in ('wind', 'dragon') :
                for a_rank in MahjongTile.WINDS_RANK + MahjongTile.DRAGONS_RANK : 
                    self.tile_set.append(MahjongTile(suit=a_suit, rank=a_rank))
                    self.tile_set.append(MahjongTile(suit=a_suit, rank=a_rank))
                    self.tile_set.append(MahjongTile(suit=a_suit, rank=a_rank))
                    self.tile_set.append(MahjongTile(suit=a_suit, rank=a_rank))
            else:
                for a_rank in range(1,10) : 
                    self.tile_set.append(MahjongTile(suit=a_suit, rank=a_rank))
                    self.tile_set.append(MahjongTile(suit=a_suit, rank=a_rank))
                    self.tile_set.append(MahjongTile(suit=a_suit, rank=a_rank))
                    self.tile_set.append(MahjongTile(suit=a_suit, rank=a_rank))
        self.yama = None
        self.players = [MahjongPlayer(f'Player {_}') for _ in range(4)]
        self.chicha = None
        self.host = self.chicha
        self.player_inturn = self.chicha
        self.wanpai_filpped = list()
        
    
    def fisher_yates_shuffle(self):
        n = len(self.yama)
        for i in range(n - 1, 0, -1):
            j = random.randint(0, i)
            self.yama[i], self.yama[j] = self.yama[j], self.yama[i]

    def shiepai(self, seed = None):
        if seed is None :
            seed = Crypto.Random.random.getrandbits(48)
        else:
            seed = int(seed)
        random.seed(seed)
        print(seed)
        self.yama = list(range(0,len(self.tile_set)))
        random.shuffle(self.yama)
        self.fisher_yates_shuffle()
        # random.shuffle(self.yama)
    
    def kaimen(self):
        ''' the elements are in clockwise order.'''
        if self.chicha is None :
            dice_roll0 = random.randint(1, 6)
            dice_roll1 = random.randint(1, 6)
            self.chicha = ((dice_roll0 + dice_roll1) -  1) % 4
            self.players = self.players[self.chicha:] + self.players[:self.chicha]
            self.chicha = 0
            self.host = self.chicha
        dice_roll0 = random.randint(1, 6)
        dice_roll1 = random.randint(1, 6)
        gate_ix = (self.host * 34 +dice_roll0 + dice_roll1 - 1) % len(self.yama)
        self.yama = self.yama[gate_ix * 2:] + self.yama[:gate_ix * 2]
        self.wanpai_filpped = list()
        self.wanpai_filpped.append(self.wanpai_index(-4))
    
    def haipai(self):
        for i in range(13*4 + 1) :
            self.players[(self.host+i)%4].hand.append(self.yama.pop(0))
        for i in range(len(self.players)):
            #pass
            self.players[i].hand.sort()
    
    '''from the last top yama[-2], yama[-1], yama[-4], yama[-3], yama[-6] = the 1st dora indicator, ...''' 
    def wanpai_index(self, ix):
        return -(ix//2)+(ix%2)-2
    
    def __str__(self):
        t = ''
        for i in range(len(self.players)):
            if self.host == i :
                t += '*'
            else:
                t += ' '
            t += f'{self.players[i].name():12}'
            for p in self.players[i].hand:
                t += str(self.tile_set[p])
            t += '\n'
        for ix in self.wanpai_filpped:
            t+= str(self.tile_set[ix])
        return t

# Example usage
if __name__ == "__main__":
    tile1 = MahjongTile(suit="man", rank=5)
    tile2 = MahjongTile(suit="pin", rank=3)
    tile3 = MahjongTile(suit='wind', rank="east")

    print(tile1)  # Output: 5m
    print(tile2)  # Output: 3p
    print(tile3)  # Output: Honor(east)
    
    print(tile3.rank)
    
    tak = MahjongTable()
    for e in tak.tile_set:
        print(e, end="")
    print()
    tak.shiepai()
    print(tak.yama)
    print(tak.kaimen())
    tak.haipai()
    print(tak)
    for pl in tak.players:
        pl.clear()
    tak.host = tak.host+ 1
    tak.shiepai()
    tak.kaimen()
    tak.haipai()
    print(tak)
    for p in tak.yama:
        print(tak.tile_set[p], end = '')

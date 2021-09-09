'''
Created on 2021/09/09

@author: sin
'''

class DFA:
    '''
    classdocs
    '''    

    def __init__(self, states, alphabet, transition, initial, finals):
        '''
        Constructor
        '''
        self._states = set(states)
        self._alphabet = set(alphabet)
        if isinstance(transition, dict) :
            self._transition = transition
        else:
            self._transition = dict()
            self.define_transition(transition)
        if initial != None and initial in self._states :
            self._initial = initial
        else:
            self._initial = 0
        self._finals = set(finals)
    
    def define_transition(self, deltastr):
        for t in deltastr.split(','):
            if t[0] in self._states and t[1] in self._alphabet :
                if len(t[2]) == 1 :
                    if t[2] in self._states :
                        self._transition[(t[0], t[1])] = t[2]
                    else:
                        raise ValueError(str(t[2])+", undefined destination state.")
                else:
                    self._transition[(t[0], t[1])] = set(t[2])
        
    def get_transition(self):
        return self._transition
    
    transition = property(get_transition)
    
    def get_states(self):
        return self._states
    
    states = property(get_states)
    
    def get_alphabet(self):
        return self._alphabet
    
    alphabet = property(get_alphabet)
    
    def get_initial(self):
        return self._initial
    
    initialstate = property(get_initial)

    def get_finals(self):
        return self._finals
    
    finalstates = property(get_finals)

if __name__ == "__main__" :
    dfa = DFA("0123", "ab", "0a1,0b0,0c3,1a1,1b2,1c3,2a1,2b2,2c3", 0, {2})
    print("hello.")
    print(dfa.states, dfa.alphabet, dfa.transition, dfa.initialstate, dfa.finalstates)

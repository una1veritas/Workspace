//
//  main.cpp
//  BitSet
//
//  Created by Sin Shimozono on 2019/05/17.
//  Copyright © 2019 Sin Shimozono. All rights reserved.
//

#include <iostream>
#include <cstdlib>

class BitSet {
    unsigned char * bits;
    unsigned int nbits;
public:
    unsigned int bytecount(unsigned int n) {
        return (n>>3) + ((n & 0x07) ? 1 : 0);
    }
    
public:
    BitSet(unsigned int n) {
        nbits = n;
        bits = new unsigned char[bytecount(nbits)];
    }
    
    ~BitSet() {
        delete [] bits;
    }
    
    unsigned int size(void) const { return nbits; }
    
    int set(unsigned int e) { bits[e>>3] |= (1<<(e & 0x07)); return 1; }
    int clear(unsigned int e) { bits[e>>3] &= ~(1<<(e & 0x07)); return 0; }
    int let(unsigned int e, int bval) {
        if (bval) return set(e); else return clear(bval);
    }
    int check(unsigned int e) const { return (bits[e>>3] & (1<<(e & 0x07))) != 0; }
    
    friend std::ostream & operator<<(std::ostream & ostr, const BitSet & bset) {
        ostr << "bset{";
        for (unsigned int e = 0; e < bset.nbits; ++e) {
            if (bset.check(e)) ostr << e << ", ";
        }
        ostr << "} ";
        return ostr;
    }
};

int main(int argc, const char * argv[]) {
    std::cout << "Hello, World!\n";
    
    BitSet bset(19);
    for (unsigned int i = 0; i < bset.size(); ++i) {
        std::cout << i << ": " << bset.bytecount(i) << std::endl;
    }
    bset.set(0);
    bset.set(2);
    bset.set(7);
    bset.set(8);
    bset.set(17);
    bset.set(16);
    bset.set(18);
    std::cout << bset << std::endl;
    return EXIT_SUCCESS;
}

//
// cardset.cc - トランプカードの集合型(C++版)
//	作者: (あなたの名前); 日付: (完成した日付)
//

#include <iostream>

#include "Card.h"
#include "CardSet.h"


void CardSet::shuffle(const time_t & seed) {
    int t, i, j, k;
    Card swap;
    
    srandom(seed);

    for (t = 0; t < 100; t++) {
      i = random() % numcard;
      j = i + (random() % (numcard-i));
      for (k = 0; i + k < j; k++) {
    	  swap = cards[k];
    	  cards[k] = cards[j-k - 1];
    	  cards[j-k - 1] = swap;
       }
    }
}


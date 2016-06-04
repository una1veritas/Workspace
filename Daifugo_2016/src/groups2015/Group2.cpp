
/*
 *  Group2.cpp

 *
 */
#include <groups2015/Group2.h>
#include <iostream>
#include <string>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"

using namespace grp2015;

void Group2::ready() {
	memory.makeEmpty(); //memory.clear();
	trump.makeEmpty(); //trump.clear();
}

bool Group2::approve(const GameStatus &gs){
  memory.insert(gs.pile);
  /*
  i = 0;
  for(i = gs){
  }
  */
  return true;

}

bool Group2::follow(const GameStatus & gstat, CardSet & s) {
	CardSet pile(gstat.pile);
	Card tmp, p;
	// int pos = 0, num = 0, j, k = 0, m = 0;
	//	std::cout << gstat << std::endl;
	s.makeEmpty(); //clear();
	sortInHand();
	//	std::cout << "( " << inHand() << " )" << std::endl;
	strategy(inHand(),s,pile);

	/*
	if(pile.size() > 0){
	  if(pile.size() >= 2){
	    if(pile[k].isJoker()){
	      k++;
	    }
	  }
	  while(pos != -1){
	    pos = findpos(pile[k], inHand(), pos);
	    num = findnum(pos,inHand());
	    if(num >= pile.size() || pos == -1){
	      break;
	    }
	    pos += num;
	  }
	  if(inHand()[pos] == inHand()[inHand().size - 1] && kindofNum(inHand())>= 3){
	    num = 0;
	  }
	  if(pile.size() == num){
	    for(j = 0; j < num; j++){
	      inHand().pickup(tmp,pos);
	      s.insert(tmp);
	    }
	  }else if(pile.size() < num){
	    for(j = 0; j < pile.size(); j++){
	      inHand().pickup(tmp,pos);
	      s.insert(tmp);
	    }
	  }
	}else{
	  num = findnum(0,inHand());
	  for(j = 0; j < num; j++){
	    inHand().pickup(tmp, 0);
	    s.insert(tmp);
	  }
	}
*/
      	//inHand().pickup(tmp, -1); // とにかく選ぶ．
	//s.insert(tmp);
	// the card idential to tmp is already removed from the hand. 
	// cardSetOfSameRanks(s, pile.size());
	// たとえば、複数枚のカードを探す関数。ただしこの関数は未実装。
	// 現状ではこの follow は Player.cc のものと等価
	return true;
}

/*
 * ソートに使う順序関数の例．
 * 自分のならべたい順序の定義を作成する．
 * これはカード同士の比較であり，カードの集合同士の比較ではない．
 */
bool Group2::cardLessThan(const Card & c1, const Card & c2) {
	if ( c1.getNumber() < c2.getNumber() )
		return true;
	return false;
}

/*
 * 順序関数と整合性のある同値（等しい）関係．
 */
bool Group2::cardEquals(const Card & c1, const Card & c2) {
	return !cardLessThan(c1, c2) && !cardLessThan(c2, c2);
}

/*
 * 順序関係 compareCards を使うナイーヴな naive ソートの例．
 * 枚数は少ないので，効率は気にしない．
 */
void Group2::sortInHand(void) {
	for(int i = 0; i+1 < hand.size(); i++) {
		for(int j = i+1; j < hand.size(); j++) {
		  if (cardGreaterThan(hand[i], hand[j])) {
				Card t = hand[i];
				hand[i] = hand[j];
				hand[j] = t;
			}
		}
	}
}

int Group2::findpos(Card & pile, CardSet & inHand, int i){
  for( ; i < inHand.size(); i++){
    if(cardGreaterThan(inHand[i], pile)){
      return i;
    }
  }
  return -1;    
}

int Group2::findnum(int pos,CardSet & inHand){
  int n;
  int i = 1;
  if(pos == -1){
    return 0;
  }
  n = inHand[pos].getNumber();
  for(pos += 1; pos < inHand.size(); pos++){
    if(inHand[pos].getNumber() == n){
      i++;
    }else{
      break;
    }
  }
  return i;
}

int Group2::kindofNum(CardSet & inHand){
  int i = 0, pos = 0, num = 0;
  while(pos != -1){
    num = findnum(pos,inHand);
    if(pos != -1){
      break;
    }
    i++;
    pos += num;
  }	
  return i;
}

void Group2::strategy(CardSet &inHand, CardSet &s, CardSet &pile){
  Card tmp;
  int pos = 0, num = 0, j, k = 0, f = 0;


  if(pile.size() > 0){
    if(pile.size() >= 2){
      if(pile[k].isJoker()){
	k++;
      }
    }
    while(pos != -1){
      pos = findpos(pile[k], inHand, pos);
      num = findnum(pos,inHand);
      if(inHand[inHand.size() - 1].isJoker() && inHand.size() <= 5){
	if(pile.size() == num + 1){
	  f = 1;
	  break;
	}
      }
      if(num >= pile.size() || pos == -1){
	break;
      }
      pos += num;
    }
    
    if(inHand[pos].getRank() == inHand[inHand.size() - 1].getRank() && kindofNum(inHand) >= 3 && f == 0){
      num = 0;
    }
    
    if(pile.size() == num){
      for(j = 0; j < pile.size(); j++){
	inHand.pickup(tmp,pos);
	s.insert(tmp);
      }
    }else if(pile.size() < num){
      for(j = 0; j < pile.size(); j++){
	inHand.pickup(tmp,pos);
	s.insert(tmp);
      }
    }else if(pile.size() < num && f == 1){
      for(j = 0; j < num; j++){
	if(f == 1){
	  inHand.pickup(tmp, inHand.size() - 1);
	  s.insert(tmp);
	  f = 0;
	}
	inHand.pickup(tmp,pos);
	s.insert(tmp);
      }
    }
  }else{
    num = findnum(0,inHand);
    for(j = 0; j < num; j++){
      inHand.pickup(tmp, 0);
      s.insert(tmp);
    }
  }


}


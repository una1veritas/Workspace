#include <groups2015/Group5.h>
#include <iostream>
#include <string>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"
using namespace grp2015;

void Group5::ready(){
	memory.makeEmpty(); //memory.clear();
	trump.makeEmpty(); //trump.clear();
	trump.setupDeck();//
}

bool Group5::follow(const GameStatus & gstat, CardSet & s){
  CardSet pile(gstat.pile);//今場のカード１〜４枚を取得CardSet型
  Card tmp[4];
  //CardSet nopare, twopare, threepare;
  int pilenum = 1;//こちらから出す数
  int pos = 0;
  int leaderflag = 0;
  s.makeEmpty();
  sortInHand();
  pilenum = pile.size();
  //std::cout << "( " << inHand() << " )" << std::endl;
  //std::cout << "( " << pile << " )" << std::endl;
   //if(approve(gstat))
     //std::cout << "( " << trump << " )" << std::endl;
   if(inHand().size() == 3){//親ではなくかつ最終局面の時
     if(is2PareinHand(hand,pos))
		pilenum = 2;
	else
		pilenum = 1;
	}
   /*for(int i = 0; i < hand.size() - 1; i++)
     if(cardEquals(hand[i],hand[i+1]))
          std::cout << "(((( (" << hand[i] << ") ))))" << std::endl;
   */
   /*if(cardLessThan(hand[i + 1],hand[i]))
         std::cout << "(((( (" << hand[i] << ") ))))" << std::endl;
   */
    if(getID() == gstat.playOrder[gstat.leaderIndex] ){//親の時に目覚め
      // std::cout << "((( " << pilenum << " )))" << std::endl;
   if(is4PareinHand(hand,pos))
       pilenum = 4;
   else if(is3PareinHand(hand,pos))
       pilenum = 3;//親ならできる限り3枚だし
   else if(is2PareinHand(hand,pos))
       pilenum = 2;
   else{
     leaderflag = 1;
       pilenum = 1;
   }	
       }
    // std::cout << "( " << pilenum << " )" << std::endl;
    // std::cout << "()(( " << pos << " )()" << std::endl;
   switch(pilenum){//
   case 0:
   case 1:{
     if(leaderflag == 1){
        inHand().pickup(tmp[0], hand.size()-1);
        s.insert(tmp[0]);  
	leaderflag = 0;
     }else{
        inHand().pickup(tmp[0], getPosbyPile(pile));
        s.insert(tmp[0]);
     }}
       break;
   case 2:{
     if(is2PareinHand(hand,pos)){
	 inHand().pickup(tmp[0], pos);
         inHand().pickup(tmp[1], pos + 1);
	 s.insert(tmp[0]);
	 s.insert(tmp[1]);
     }}
       break;
   case 3:{
     if(is3PareinHand(hand,pos)){   
	  inHand().pickup(tmp[0], pos);
          inHand().pickup(tmp[1], pos + 1);
          inHand().pickup(tmp[2], pos + 2);
	   s.insert(tmp[0]);
	   s.insert(tmp[1]);
	   s.insert(tmp[2]);
	  }
   }
       break;
   case 4:{
	for(int i = 0; i < 4; i++){
		inHand().pickup(tmp[i],pos + i);
	}
	for(int i= 0; i < 4; i++){
		s.insert(tmp[i]);	
	}
   }	
	break;
}
   // std::cout << "( " << tmp[0] << " )" << std::endl;
   //std::cout << "( " << tmp[1] << " )" << std::endl;
   //std::cout << "( " << tmp[2] << " )" << std::endl;
   //std::cout << "( " << tmp[3] << " )" << std::endl;
  return true;
}
bool Group5::is2PareinHand(CardSet & s, int & pos){
  for(int i = 0; i < s.size() - 1; i++){
    if(cardEquals(hand[i], hand[i+1])){
      pos = i;
	return true;
    }
  }
 return false;
}
bool Group5::is3PareinHand(CardSet & s, int & pos){
 for(int i = 0; i < s.size() - 2; i++){
   if(cardEquals(hand[i],hand[i+1]) && cardEquals(hand[i+1],hand[i+2])){
           pos = i;
		return true;
   }
 }
 return false;
 }
bool Group5::is4PareinHand(CardSet & s, int & pos){
 for(int i = 0; i < s.size() - 3; i++){
   //std::cout << "(((( (" << s.size() << ") ))))" << std::endl;
   if(cardEquals(hand[i],hand[i + 1]) && cardEquals(hand[i + 1],hand[i + 2]))
     if(cardEquals(hand[i + 2],hand[i + 3])){
	  std::cout << "(((( (" << hand[i] << ") ))))" << std::endl;
	  std::cout << "(((( (" << hand[i+1] << ") ))))" << std::endl;
          std::cout << "(((( (" << hand[i + 2] << ") ))))" << std::endl;
          std::cout << "(((( (" << hand[i + 3] << ") ))))" << std::endl;
	  pos = i;
	  return true;
     }
}
return false;
}
int Group5::getPosbyPile(CardSet & pile){
  Card tmp;
  int pos;
  pos = -1;
  if(!pile.isEmpty()){
  pile.pickup(tmp,-1);
  for(int i = 0; i < hand.size(); i++)
    if(cardLessThan(tmp, hand[i]))
      pos = i;
  }
  return pos;
}
/*void n231054s::makeparebyHand(CardSet & s){
  Card tmp, tmp2, tmp3;
  }*/
bool Group5::cardLessThan(const Card & c1, const Card & c2) {
  int i,j;
  i = c1.getNumber();
  j = c2.getNumber();
  if(i < 3 && 0 < i)
    i += 13;
  if(j < 3 && 0 < j)
    j += 13;
  if ( i < j )
     return true;

  return false;
}
bool Group5::cardEquals(const Card & c1, const Card & c2) {
	return !cardLessThan(c1, c2) && !cardLessThan(c2, c1);
}
void Group5::sortInHand(void) {
	for(int i = 0; i+1 < hand.size(); i++) {
		for(int j = i+1; j < hand.size(); j++) {
			if ( cardLessThan(hand[i], hand[j]) ) {
				Card t = hand[i];
				hand[i] = hand[j];
				hand[j] = t;
			}
		}
	}
}
 void Group5::sortInTrump(void){
   for(int i = 0; i+1< trump.size(); i++){
     for(int j = i + 1; j < trump.size(); j++){
       if(cardLessThan(trump[i], trump[j])){
	 Card t = trump[i];
	 trump[i] = trump[j];
	 trump[j] = t;
       }
     }
   }
 }
bool Group5::approve(const GameStatus & gstat){
  int MenbersCardNum;//自分意外のメンバーのカード枚数
  CardSet pile(gstat.pile);
  sortInTrump();
  for(int i = 0; i < pile.size(); i++)
    trump.remove(pile[i]);
  for(int i = 0; i < gstat.numPlayers; i++)
    MenbersCardNum += gstat.numCards[i];
  return true;
}

/*
 *  SamplePlayer.cpp
 *  PlayingCard
 *
 *  Created by 下薗 真一 on 09/04/12.
 *  Modified by Kazutaka Shimada on 09/04/21.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include <string>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"

#include "Group3.h"


void Group3::ready() {
	// 最初にカードを配られた状態
	mymemory.makeEmpty(); //memory.clear();
	sort();
	for(int i = 0; i < 16; i++)
	  CardNum2[i] = 4;
	CardNum2[16] = 1;
	before2 = 53;
}




bool Group3::follow(const GameStatus & gstat, CardSet & s) {
	CardSet pile(gstat.pile);
	CardSet mycards;
	Card card;
	/*
      	printf("カードの強さの平均 = %lf\n", allaverage());
	printf("最も弱いカードの番号 = %d\n", weakest());
	if(strongest(gstat)==0){
	  printf("\nnow strongest card is jkr \n");
	}
	else{
	  printf("\nnow strongest number is %d\n",strongest(gstat));
	}
	
	std::cout << std::endl << gstat << std::flush;
      	sort();
//	std::cout << "average: " << average(hand) << std::endl;
	std::cout << inHand() << std::endl;
	*/
	int pilesize;
	int kind;
	//	int strength;
	//findSmallestAcceptable(pile, mycards);

        kind = kindcount();
	pilesize = pile.size();

	mycardcount();

	int j;
	if(pilesize != 0){
	  for(int i = 0; i < hand.size(); i++){ // 場にカードがある時

	    if (cardStrength(pile) < cardStrength(hand[i])){  // 場のカードより強く、手札の中で弱いカードを探す
	      for(j = 1 ; i+j < hand.size() ;j++ ){ // 取れる枚数全部取ってくる 
		if(cardStrength(hand[i]) != cardStrength(hand[i + j]))
		  break;
	      }

	      if(j == pilesize){//　同じ枚数取ってこれたなら
		int Number = hand[i].getNumber();
		if(Number == 1)
		  Number = 14;
		if(Number == 2)
		  Number = 15;
		if(Number == 0)
		  Number = 16;
		CardNum2[Number] -= pilesize;/*サイズ分減らす*/
		before2 -= pilesize;
		after2 = before2;
		for(int k = 0;k < pilesize;k++){
		  s.insert(hand[i+k]);
		}
		hand.removeAll(s);
		return true;
	      }

	      if(j > pilesize){
		int Number = hand[i].getNumber();
		if(Number == 1)
		  Number = 14;
		if(Number == 2)
		  Number = 15;
		if(Number == 0)
		  Number = 16;
		if(getlead(Number,pilesize)){ // 親が取れるなら
		  CardNum2[Number] -= pilesize;/*サイズ分減らす*/
		  before2 -= pilesize;
		  after2 = before2;
		  for(int k = 0;k < pilesize;k++){
		    s.insert(hand[i+k]);
		  }
		  hand.removeAll(s);
		  return true;
		}else{
		  i = i + j - 1;
		}
		
	      }
	      
	      
	    }
	      
	      
	  }
	  
	}	

	else{ //　場にカードがない時
	  /*
	  int Number = hand[0].getNumber();
	  j=0;
	  

	  if(kind >= 2){
	    if(kind == 2){ 

	      for(int i=3; i<=16;i++){
		if(getlead(i,myCard[i])){
		  Number = i;
		  break;
		}
	      }
	    }else if(kind == ) {
	      
	    } 
	    
	  }   
	  for(j = 0; j < hand.size(); j++)// 手札の最弱のカードを出せる枚数出す
	    if(hand[j].getNumber() == Number)
	      s.insert(hand[j]);
	  
	  
	  hand.removeAll(s);
	  
	  
	  if(Number == 1)
	    Number = 14;
	  if(Number == 2)
	    Number = 15;
	  if(Number == 0)
	    Number = 16;
	  CardNum2[Number] -= s.size();
	  before2 -= s.size();
	  after2 = before2;
	  
	  return true;
	  */
	  
	  int Number = hand[0].getNumber();
	  j=0;


	  if(strongest(gstat) == hand[hand.size()-1].getNumber() && Number != strongest(gstat)){ // 最強を持っていて２種類以上の時自分の最弱から２番目を出す

	    while(hand[j].getNumber() == Number ){
	      j++;
	    }
	    Number = hand[j].getNumber();
	  }
	  for(; j < inHand().size() && hand[j].getNumber() == Number; j++)// 手札の最弱のカードを出せる枚数出す
	    s.insert(hand[j]);
	 
	  
	  hand.removeAll(s);

	  
	  if(Number == 1)
	    Number = 14;
	  if(Number == 2)
	    Number = 15;
	  if(Number == 0)
	    Number = 16;
	  CardNum2[Number] -= j;
	  before2 -= j;
	  after2 = before2;
	  
	  return true;


	}
	


	  //	  s.insert(hand[0]);
	  //s.makeEmpty();
	
	
	return true;
	
	/*
	hand.pickup(card, -1); // とにかく一枚選ぶ．
	std::cout << "try " << card << std::endl;
		s.insert(card);
		if ( cardsStrongerThan(s, pile) ) {
			return true;
		} 
		else {
			// やっぱやめる
			hand.insertAll(s);
			s.makeEmpty();
			return true;
		}
*/


	
	}

	
bool Group3::approve(const GameStatus & gstat) {
  /*
    自分の番が最初の時、approveが呼び出されていないため枚数を減らすことができていない。
　　　きちんとパスができるプログラムが組めればカードを出すプログラム内で枚数を減らすことができそう
　　　現在複数枚出したときも減るように対応している。
   */
  //  printf("approveが呼び出されました\n");
  after2 = 0;
  for(int i = 0; i < gstat.numPlayers ; i++)
    after2 += gstat.numCards[i];/*全員の手札の合計を調べる*/
  //  printf("after2 = %d, before2 = %d\n", after2, before2);
  if(after2 < before2){/*全員の手札の総数に変化があったとき*/
    for(int k = 0; k < gstat.pile.size(); k++){
      int Number = gstat.pile[k].getNumber();/*場に出ているカードの番号*/
      //      printf("場に出ているカードの番号は%dです。\n", Number);
      if(Number == 1)
	Number = 14;
      if(Number == 2)
	Number = 15;
      if(gstat.pile[k].isJoker())
	Number = 16;
      CardNum2[Number] -= 1;/*一枚減らす*/
    }
  }
  /* デバッグ用
  for(int j = 3; j < 17; j++)
    printf("CardNum2[%d] = %d\n", j, CardNum2[j]);
  */
  before2 = after2;

  return true;
}

double Group3::allaverage(){
  double ave = 0;
  double num = 0;
  for(int i = 3; i < 17; i++){
    ave += CardNum2[i] * i;
    num += CardNum2[i];
  }
  ave = ave/num;

  return ave;
  
}

int Group3::strongest(const GameStatus & gstat){
  int i;
  for(i=16;i>2;i--){
    if(CardNum2[i]!=0){
      if(i==16)
	return 0;
      else if(i==15)
	return 2;
      else if(i ==14)
	return 1;
      
      return i;
    }
  }   
      
  return -1;
}

int Group3::weakest(){//まだ出ていないカードの中で最も弱いカード
  for(int i = 3; i < 16; i++){
    if(CardNum2[i] > 0){
      if(i < 14)
	return i;
      else if(i == 14)
	return 1;
      else if(i == 15)
	return 2;
      else
	return 0;//Joker
    }
  }
  return -1;
}


int Group3::kindcount(){
  int i,kindnum,tmpnum;
  tmpnum = -1;
  kindnum = 0;
  for(i=0;i<hand.size();i++){
    if(hand[i].getNumber()!=tmpnum){
      tmpnum = hand[i].getNumber();
      kindnum++;
    }
  }
  return kindnum;
}	

bool Group3::getlead(int num, int size){ // numが数字 sizeが枚数

  int enemycard;

  for(int i = num+1; i <= 16; i++){
    enemycard = CardNum2[i] - myCard[i];
    if(enemycard >= size)
      return false;
  }
  return true;
}


void Group3::mycardcount(){

  for(int i = 3; i<17; i++)
    myCard[i] = 0;

  int n;

  for(int i = 0; i < hand.size(); i++){
    if(hand[i].getNumber() >=3 )
      n = hand[i].getNumber();
    else if(hand[i].getNumber() == 0)
      n = 16;
    else if(hand[i].getNumber() == 1)
      n = 14;
    else if(hand[i].getNumber() == 2)
      n = 15;
    myCard[n]++;
  }
}



/*
 * 順序関係 compareCards を使うナイーヴ naive なソート．
 * 枚数は少ないので，効率は気にしない．
 */
void Group3::sort(bool ascending) {
	for(int i = 0; i+1 < hand.size(); i++) {
		for(int j = i+1; j < hand.size(); j++) {
			if ( (ascending && cardGreaterThan(hand[i], hand[j]))
					|| (!ascending && cardLessThan(hand[i], hand[j])) ) {
				Card t = hand[i];
				hand[i] = hand[j];
				hand[j] = t;
			}
		}
	}
}

//   Returns true if and only if the left CardSet is either correct and stronger than
// the right one, or the right one is an illegal set.
//
bool Group3::cardsStrongerThan(const CardSet & left, const CardSet & right) {
	int leftRank, rightRank;

	// regarded as "pass"
	if (left.isEmpty() )
		return false;

	// left is an illegal set
	if (!checkRankUniqueness(left))
		return false;
	if ( left.size() >= 5 )
		return true;

	// left always wins
	if ( right.isEmpty() )
		return true;

	// right is an illegal set.
	if (!checkRankUniqueness(right))
		return true;
	if ( right.size() >= 5 )
		return true;

	// the number of cards of the left set must be match to that of the right one.
	if ( left.size() != right.size() )
		return false;


	leftRank = cardStrength(left);
	rightRank = cardStrength(right);

	if ( leftRank > rightRank )
		return true;
	else
		return false;
}

int Group3::cardStrength(const CardSet & cs) {
	int i;
	if ( cs.isEmpty() )
		return 0;

	if ( cs.size() == 1 && cs[0].isJoker() ) {
		return cardStrength(cs[0]);
	}
  	for (i = 0; i < cs.size(); i++) {
	  if (!cs[i].isJoker()) {
		  break;
	  }
	}
	return cardStrength(cs[i]);
}

bool Group3::checkRankUniqueness(const CardSet & cs) {
	int rank = 0;

	if (cs.size() == 0)
		return false;

	if ( cs.size() == 1 && cs[0].isJoker() )
		return true;

	for (int i = 0; i < cs.size(); i++) {
	  if (cs[i].isJoker() )
		  continue;  // Jkrをスキップ
	  if ( rank == 0 ) {
		  rank = cs[i].getNumber();
	  } else if ( rank != cs[i].getNumber() ) {
	    return false;
	  }
	}
	return true;
}

double Group3::average(const CardSet & cset) {
	double sum = 0;
	if ( cset.size() == 0 )
		return sum;
	for(int i = 0; i < cset.size(); i++) {
		sum += cardStrength(cset[i]);
	}
	return sum/cset.size();
}

int Group3::cardStrength(const Card & c) {
	if ( c.isJoker() )
		return 18;
  	if ( c.getNumber() <= 2 )
  		return c.getNumber() + 13;
	return c.getNumber();
}

CardSet & Group3::findSmallestAcceptable(const CardSet & cs, CardSet & mycs) {
	// assumes the hand is sorted in ascending order
	mycs.makeEmpty();
	int cssize = cs.size();
	int csstrength = cardStrength(cs);
	int i, n;
	if ( cssize > 1 )
		std::cout << "multiple cards!!!" << std::endl;
	for(i = 0; i < hand.size(); ) {
		if ( cardStrength(hand[i]) <= csstrength ) {
			++i;
			continue;
		}
		for(n = 1; i + n < hand.size(); ) {
			if ( hand[i+n].isJoker()
				|| (cardStrength(hand[i + n]) == cardStrength(hand[i])) ) {
				++n;
				continue;
			}
			break;
		}
		if ( cssize <= n ) {
			if ( cssize != 0 )
				n = cssize;
			for(int j = i; j < i + n; j++)
				mycs.insert(hand[j]);
			return mycs;
		}
		i += n;
	}
	return mycs;   // empty card set.
}

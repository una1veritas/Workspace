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

#include "Group2.h"

void Group2::ready() {
	// 最初にカードを配られた状態
	mymemory.makeEmpty(); //memory.clear();
	sort();
}
bool Group2::approve(const GameStatus & gstat){
  	CardSet pile(gstat.pile);
	mymemory.insert(pile);
	sort();
	return true;
}
bool Group2::follow(const GameStatus & gstat, CardSet & s) {
	CardSet pile(gstat.pile);
	CardSet mycards;
	Card card,tmp;
//	std::cout << std::endl << gstat << std::flush;
	sort();
//	std::cout << "average: " << average(hand) << std::endl;
//	std::cout << inHand() << std::endl;
	int num,num1,a,b,c,d,e,i,j,k;
	e=hand.size();
	sort();
	if(pile.isEmpty()){
	  if(hand.size()>5){
	    num1=hand[0].getNumber();
	    inHand().pickup(tmp,0);
	    s.insert(tmp);
	    for(i=0;num1==hand[i].getNumber();){
	      if(!hand.isEmpty()){
		hand.pickup(tmp,i);
		s.insert(tmp);
	      }
	    }
	    return true;
	  }
	  else{
	    num=1;
	    k=hand[hand.size()-1].getNumber();
	    for(i=0;i<hand.size()-1;i++){
	      for(j=1;j<4;j++){
		if( i+j < inHand().size() &&  hand[i].getNumber()==hand[i+j].getNumber()){
		  num++;
		}
	      }
	      num1=hand[i].getNumber();
	      if(num>1&&k>hand[i].getNumber()){
		for(j=0;num1==hand[i].getNumber();){
		  if(!hand.isEmpty()){
		    hand.pickup(tmp,i);
		    s.insert(tmp);
		  }
		}
		return true;
	      }
	      if(hand.size()==num){
		for(j=0;j<=num;j++){
		  if(!hand.isEmpty()){
		    hand.pickup(tmp,i);
		    s.insert(tmp);
		  }
		}
		return true;
	      }
	    }
	    hand.pickup(tmp,0);
	    s.insert(tmp);
	    return true;
	  }
	}
	a=1;

	d=mymemory.size();
//	for(i=1;i<4;i++){
	for(i=1;i<pile.size();i++){
	  if(pile[0].getNumber()==pile[i].getNumber())
	    a++;
	}
	b=pile[0].getNumber();
	if(b==1||b==2){
	  b=b+13;
	}

	for(i=0;i<hand.size();i++){
	  c=hand[i].getNumber();
	  num=1;
	  k=i;
	  if(c!=0){
	    for(j=i; j+1 < hand.size() && c==hand[j+1].getNumber();j++){
	      if(j<hand.size()-1){
		num++;
	      }
	    }
	    if(c==1||c==2){
	      c=c+13;
	    }
	    if(d<20){
	      if(c<14){
		if(num==a && a==1 && c>b){
		  inHand().pickup(tmp,i);
		  s.insert(tmp);
		  return true;
		}
		if(num>=a && a!=1 && c>b){
		  for(j=i;j<i+a;j++){
		    hand.pickup(tmp,i);
		    s.insert(tmp);
		  }
		  return true;
		}
	      }
	    }
	    else{
	      if(num>=a && c>b && i+num < inHand().size() ){
		e=same(i+num);
		if(e==a&&num>a){
		  for(j=i+num;j<i+a+num;j++){
		    inHand().pickup(tmp,i+num);
		    s.insert(tmp);
		  }
		  return true;
		}
		for(j=i;j<i+a;j++){
		  inHand().pickup(tmp,i);
		  s.insert(tmp);
		}
		return true;
	      }
	      if(a==1 && c>b){
		inHand().pickup(tmp,i);
		s.insert(tmp);
		return true;
	      }
	    }
	  }
	  else{
	    if(b==15&&a==1&&d<20){
	      inHand().pickup(tmp,i);
	      s.insert(tmp);
	      return true;
	    }
	    else{
	      if(a==1&&d>=20){
		inHand().pickup(tmp,i);
		s.insert(tmp);
		return true;
	      }
	    }
	  }
	  i=i+num-1;
	}
	// the card identical to tmp is already removed from the hand.
	// cardSetOfSameRanks(s, pile.size());
	// たとえば、複数枚のカードを探す関数。ただしこの関数は未実装。
	// 現状ではこの follow は Player.cpp のものと等
	return true;
}
int Group2::same(int a){
  int j,num,c;
  num=1;
  c=hand[a].getNumber();
  for(j=a; j+1 < hand.size() && c==hand[j+1].getNumber();j++){
    if(j<hand.size()-1){
      num++;
    }
  }
  return num;
}
/*
 * 順序関係 compareCards を使うナイーヴ naive なソート．
 * 枚数は少ないので，効率は気にしない．
 */
void Group2::sort(bool ascending) {
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
bool Group2::cardsStrongerThan(const CardSet & left, const CardSet & right) {
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

int Group2::cardStrength(const CardSet & cs) {
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

bool Group2::checkRankUniqueness(const CardSet & cs) {
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

double Group2::average(const CardSet & cset) {
	double sum = 0;
	if ( cset.size() == 0 )
		return sum;
	for(int i = 0; i < cset.size(); i++) {
		sum += cardStrength(cset[i]);
	}
	return sum/cset.size();
}

int Group2::cardStrength(const Card & c) {
	if ( c.isJoker() )
		return 18;
  	if ( c.getNumber() <= 2 )
  		return c.getNumber() + 13;
	return c.getNumber();
}

CardSet & Group2::findSmallestAcceptable(const CardSet & cs, CardSet & mycs) {
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

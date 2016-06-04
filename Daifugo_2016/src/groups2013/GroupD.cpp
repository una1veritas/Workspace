/*
 * GroupD.cpp
 *  PlayingCard
 *
 *  Created by GroupD on 13/05/16.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include <string>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"
#include "GroupD.h"
using namespace grp2013;

#define TAIKAI_HONBAN

GroupD::GroupD(const char * s) : Player(s) {}

bool GroupD::approve(const CardSet & pile, int numCards[]) {
  memory.insert(pile);
  // memoryには他のプレイヤーが場に出したカードがストックされる 
  
#ifndef TAIKAI_HONBAN
  // 自分自身のidは getId() で獲得できる　
   std::cout << "Demo: Called GroupD::approve() for '"<< playerName()
		   << "', id = " << getID() << ". " << std::endl;

   std::cout << "Demo: Player's cards in hands: " << std::endl;
  for (int i=0; numCards[i] != 99; i++){
    // numCardsには自分を含めたそれぞれの持ち札の数が格納されている
    // 配列の最後には終端検知のため"99"という値が入っている
    // numCardsの添字はプレイヤーのidを表す
     std::cout << "ID " << i << ": " << numCards[i] << std::endl;
  }
  std::cout << std::endl;
#endif
  return true;
  
}

bool GroupD::follow(const CardSet & pile, CardSet & s) {
  int i,k = 0, /*lib,*/ address=0,count[14];
  Card tmp;
  s.makeEmpty();
  if(pile.isEmpty()){
    if(cardSetOfSameRanks(s,-1,4)){
    }
    else if(cardSetOfSameRanks(s,-1,3)){
    }
    else if(cardSetOfSameRanks(s,-1,2)){
    }
    else{

      for(i = 0;i < inHand().size(); i++){
	if( cardGreaterThan(inHand()[address], inHand()[i]) )
	  address = i;		
      }
      inHand().pickup(tmp, address); // anyway, choose a card.
      // hand は Player の private な変数なので直接さわれないことに注意
      s.insert(tmp);
      // the card idential to tmp is already removed from the hand. 
    }
  }
  else {
    if(pile.size() > 1){
      cardSetOfSameRanks(s, pile[0].getRank(), pile.size());
    }
    else{
      for(i = 0;i < 14;i++)
	count[i] = 0;
      
      for(i = 0;i < inHand().size(); i++){
	if(!inHand()[i].isJoker())
	    count[inHand()[i].getRank()]++;
      }
      for(i = 0;i < inHand().size();i++)
	if(count[inHand()[i].getRank()] == 1)
	  address = i;

      for(i = 0;i < inHand().size();i++){
       	if(count[inHand()[i].getRank()] == 1){
	  if( cardGreaterThan(inHand()[i], pile[0]) &&
	     cardGreaterThan(inHand()[address], inHand()[i]) ) {
	    address = i;
	    k = 1;
	  }
	  else if( cardGreaterThan(inHand()[i], pile[0]) &&
		  ( cardGreaterThan(pile[0], inHand()[address]) ||
		   (inHand()[address].getRank() == pile[0].getRank())) ) {
	    address = i;
	    k = 1;
	  }
	}
      }
      if(k == 0)
	for(i = 0;i < inHand().size();i++){
	  if( cardGreaterThan(inHand()[i], pile[0]) &&
	     cardGreaterThan(inHand()[address], inHand()[i]) ){
	    address = i;
	  }
	  else if( cardGreaterThan(inHand()[i], pile[0]) &&
		  cardGreaterThan(pile[0], inHand()[address]) ) {
	    address = i;
	  }
      }
      inHand().pickup(tmp, address);
 // anyway, choose a card.
      // hand は Player の private な変数なので直接さわれないことに注意
      s.insert(tmp);
      // the card idential to tmp is already removed from the hand. 
    }
  }
  
  //	cardSetOfSameRanks(s, pile.size());
  // たとえば、複数枚のカードを探す関数。ただしこの関数は未実装。
  // 現状ではこの follow は Player.cc のものと等価
  return true;
}


bool GroupD::cardSetOfSameRanks(CardSet &s,int x, int y){
  int i=0,j,k=0, address,count[14];
  Card tmp;
  for(i = 0;i < 14;i++)
    count[i] = 0;
  
  for(j = 0;j < inHand().size(); j++){
    if(!inHand()[j].isJoker())
      count[inHand()[j].getRank()]++;
  }
  
  for(i = 3;i < 14 && k == 0; i++){
    if(count[i] == y && i > x)
      k = i;
  }
  if(k > 0){
    for(address = 0; address < inHand().size(); address++){
      if(inHand()[address].getRank() == k){
	inHand().pickup(tmp, address);
	s.insert(tmp);
	address--;
      }
    }

    return true;
  }
  
  return false;
  
};



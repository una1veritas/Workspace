/*
 *  LittleThinkPlayer.cpp
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
#include "GroupB.h"
using namespace grp2013;

#define TAIKAI_HONBAN

bool GroupB::approve(const GameStatus & gstat /* , const CardSet & pile, int numCards[]*/) {
  memoryInsert(gstat.pile);
  // memoryには他のプレイヤーが場に出したカードがストックされる
  
  // 自分自身のidは getId() で獲得できる
  // std::cerr << "your information is "<< getName() << " " << getId() << std::endl;

  myCards = hand.size();

  return true;
  
}

bool GroupB::follow(const CardSet & pile, CardSet & s) {
  sort();
  searchPair();

  s.makeEmpty();

  if(pile.isEmpty()){
    // 最初の手番
    Card temp;
    do{
      // std::cerr << "check1\n";
      inHand().pickup(temp, 0);
      s.insert(temp);
      if(inHand().isEmpty()){
	break;
      }
    } while(inHand()[0].getRank() == temp.getRank());
  }
  else{
    if(pile.size() >= 2){
      multiCard(pile, s);
    }
    else{
      // 前回のカードを見る
      Card pileCard = pile[0];
#ifndef TAIKAI_HONBAN
      pileCard.print();
#endif
      
      // 弱いカードから出せるか見ていく
      for(int i=0; i < inHand().size(); i++){
	Card handCard = inHand()[i];

	if(myCards == 1 && cardGreaterThan(handCard, pileCard) ){
	  inHand().pickup(handCard, i);
	  s.insert(handCard);
	  break;
	}

	if(handCard.getRank() == handGreatestCard.getRank()){
	  if( cardGreaterThan(handCard, fieldGreatestCard) ||
	     handCard.getRank() == fieldGreatestCard.getRank()){
	    if( cardGreaterThan(handCard, pileCard) ){
	      inHand().pickup(handCard, i);
	      s.insert(handCard);
	      break;
	    }
	  }
	}
	else{
	  if( !pairFlag[i] && cardGreaterThan(handCard, pileCard) ){
	    inHand().pickup(handCard, i);
	    s.insert(handCard);
	    break;
	  }
	  if(myCards <= 6 && cardGreaterThan(handCard, pileCard) ){
	    inHand().pickup(handCard, i);
	    s.insert(handCard);
	    break;
	  }
	}
      }
    }
  }

  memoryInsert(s);

  if(inHand().isEmpty()){
    init();
  }

  return true;
}

// こいつは使ってない
bool GroupB::multiCard(CardSet & s)
{
  for(int i=0; i < inHand().size(); i++){
    if(pairFlag[i]){
      Card c;
      do{
	inHand().pickup(c, i);
	s.insert(c);
      } while(inHand()[i].getRank() == c.getRank());
      return true;
    }
  }

  return false;
}

bool GroupB::multiCard(const CardSet & pile, CardSet & s)
{
  int pileSize = pile.size();
  Card pileCard = pairRank(pile);

  CardSet temp;

  for(int i=0; i < inHand().size(); i++){
    if(pairFlag[i] && 
       cardGreaterThan(inHand()[i], pileCard) &&
       inHand()[i].getRank() != handGreatestCard.getRank()){
      Card c;
      do{
	if(i >= inHand().size()){
	  break;
	}
	inHand().pickup(c, i);
	temp.insert(c);
      } while(inHand()[i].getRank() == c.getRank());
      if(temp.size() == pileSize){
	s.insertAll(temp);
	return true;
      }
      //i += (temp.size() - 1);
      inHand().insert(temp);
      sort();
      searchPair();
    }
  }

  return false;
}

void GroupB::init()
{
  memory.makeEmpty();
  fieldGreatestCard.set(4, 0); // JOKER
}

void GroupB::sort()
{
  int size = inHand().size();
  Card* hand = new Card[size];
  for(int i=0; i < size; i++){
    inHand().pickup(hand[i], 0);
  }

  Card temp;
  for(int i=0; i < size-1; i++){
    for(int j=size-1; j > i; j--){
      if( cardGreaterThan(hand[j-1], hand[j]) ){
	temp = hand[j];
	hand[j] = hand[j-1];
	hand[j-1] = temp;
      }
    }
  }

  for(int i=0; i < size; i++){
    inHand().insert(hand[i]);
  }

  handGreatestCard = hand[size-1];
#ifndef TAIKAI_HONBAN
  // std::cerr << "\nHand Greatest Card : ";
  handGreatestCard.print();
  // std::cerr << "\n";
  // std::cerr << "\nsort : ";
  inHand().print();
  // std::cerr << "\n";
#endif

  delete [] hand;

  // std::cerr << "\nSort End\n";
}

void GroupB::searchPair()
{
  int size = inHand().size();

  if (pairFlag != NULL){
    delete [] pairFlag;
  }
  pairFlag = new bool[size];

  for(int i=0; i < size; i++){
    pairFlag[i] = false;
  }

  for(int i=0; i < size-1; i++){
    if(inHand()[i].getRank() == inHand()[i+1].getRank()){
      pairFlag[i] = pairFlag[i+1] = true;
    }
  }
}

int GroupB::searchMemory(int num)
{
  int count = 0;
  for(int i=0; i < memory.size(); i++){
    Card c = memory[i];
    if(c.getRank() == num){
      count++;
    }
  }

  return count;
}

void GroupB::memoryInsert(const CardSet& cards)
{
  memory.insert(cards);

  int flag;
  do{
    flag = false;
    if(fieldGreatestCard.isJoker() && 
       searchMemory(fieldGreatestCard.getRank()) >= 1){
      fieldGreatestCard.set(0, 2);
      flag = true;
      // std::cerr << "Joker is taked out\n";
    }
    else if(searchMemory(fieldGreatestCard.getRank()) >= 4){
      int rank = fieldGreatestCard.getRank();
      if(rank == 3){
	break;
      }
      // std::cerr << rank << " is taked out\n";
      if(--rank <= 0){
	rank = 13;
      }
      fieldGreatestCard.set(0, rank);
      flag = true;
    }
  } while(flag);

#ifndef TAIKAI_HONBAN
  // std::cerr << "\nField Greatest Card : ";
  fieldGreatestCard.print();
  // std::cerr << "\n";
#endif
}

Card GroupB::pairRank(const CardSet &pile)
{
  if(pile.size() > 1)
    if(pile[0].isJoker())
      return pile[1];

  return pile[0];
}

/*
 *  Player.h
 *  PlayingCard
 *
 *  Created by 下薗 真一 on 09/04/12.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include "Player.h"
#include "GameStatus.h"

namespace grp2013 {

class GroupB : public Player {
  CardSet memory;
  bool* pairFlag;
  int myCards;

  Card fieldGreatestCard;
  Card handGreatestCard;
  
 public:
  GroupB() : Player("Bob") {
    init();
  }

  bool follow(const CardSet &, CardSet &);
//  bool approve(const CardSet &, int[]);
  bool follow(const GameStatus & gstat, CardSet & cards) {
	  return follow(gstat.pile, cards);
  }
  bool approve(const GameStatus & gstat);

 private:
  void init();
  void sort();
  void searchPair();
  bool multiCard(CardSet &);
  bool multiCard(const CardSet &, CardSet &);
  int searchMemory(int);
  void memoryInsert(const CardSet &);
  Card pairRank(const CardSet &);
  
};

}

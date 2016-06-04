/*
 *  Player.h
 *  PlayingCard
 *
 *  Created by 下薗 真一 on 09/04/12.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
namespace grp2014 {

class GroupB : public Player {
  CardSet memory;
  bool* pairFlag;
  int myCards;

  Card fieldGreatestCard;
  Card handGreatestCard;
  
 public:
  GroupB(const char name[] = "Bob") : Player(name) {
    init();
  }

  bool follow(const CardSet &, CardSet &);
  bool approve(const CardSet &, int[]);
  virtual bool follow(const GameStatus & gstat, CardSet & cards) {
	  return follow(gstat.pile, cards);
  }
  virtual bool approve(const GameStatus & gstat) {
	  return approve(gstat.pile, (int *) gstat.numCards);
  }
  
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

/*
 *  Player.h
 *  PlayingCard
 *
 *  Created by 下薗 真一 on 09/04/12.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

namespace grp2013 {
class GroupG : public Player {
	CardSet memory;
	CardSet restCards;

	// 追加メンバ変数
	int myCardSize;
	int *rivalCardSize;

public:
	GroupG(const char * name = "George");

	bool follow(const CardSet &, CardSet &);
	bool approve(const CardSet &, int[]);
	bool follow(const GameStatus & gstat, CardSet & cards) {
	  return follow(gstat.pile, cards);
	}
	bool approve(const GameStatus & gstat) {
	  return approve(gstat.pile, (int *) gstat.numCards);
	}


	// 追加メンバ関数
	void removeCards(CardSet &, CardSet);

	void sort(CardSet &);
	void printDetail();

	void think(CardSet, int &, int &);
	int getRevisedRank(int);
	int getGreater(Card, int);
	int getSizeOfSameRankCard(CardSet, int);
	void setRankCombination(int, int, int);
	int getMinCardSize();

	void sendCard(int, int, CardSet &);
};
}

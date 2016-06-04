/*
 *  Player.h
 *  PlayingCard
 *
 *  Created by 下薗 真一 on 09/04/12.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

namespace grp2014 {

class GroupC : public Player {
  CardSet memory;

public:
	GroupC(const char name[] = "Charlie") : Player(name) {}
	bool follow(const CardSet &, CardSet &);
	bool approve(const CardSet &, int[]);
	
	bool follow(const GameStatus & gstat, CardSet & cards) {
		return follow(gstat.pile, cards);
	}
	bool approve(const GameStatus & gstat) {
		return approve(gstat.pile, (int *) gstat.numCards);
	}

};

}


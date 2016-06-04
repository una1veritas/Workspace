/*
 *  Player.h
 *  PlayingCard
 *
 *  Created by  GroupD on 13/05/16.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

namespace grp2013 {

class GroupD : public Player {
  CardSet memory;

public:
  GroupD(const char * myname );
	bool follow(const CardSet &, CardSet &);
	bool approve(const CardSet &, int[]);
	bool cardSetOfSameRanks(CardSet &, int, int);

	bool follow(const GameStatus & gstat, CardSet & cards) {
	  return follow(gstat.pile, cards);
	}
	bool approve(const GameStatus & gstat) {
	  return approve(gstat.pile, (int *) gstat.numCards);
	}

};
}

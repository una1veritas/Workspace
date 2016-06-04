/*
 *  ThinkTA.h
 *  PlayingCard
 *
 *  Created by TA (Ryosuke Tadano) @ 2009
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef _THINKTA_H_
#define _THINKTA_H_

#include "Player.h"

class ThinkTA1 : public Player {

public:
	ThinkTA1(const char * name = "TA1") : Player(name) {}

	virtual bool follow(const GameStatus & gstat, CardSet & cards);

	bool sort(CardSet&);
};

#endif

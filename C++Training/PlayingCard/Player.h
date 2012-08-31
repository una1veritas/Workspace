/*
 *  Player.h
 *  PlayingCard
 *
 *  Created by on 09/04/12.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

class Player {
	CardSet hand;
	std::string name;
	
public:
	Player(char *);
	Player(std::string);
	
	std::string printString(void) const;
	void clearHand();
	bool isEmptyHanded();
	bool pickup(Card );
	bool takeCards(CardSet &);
	
	CardSet & inHand() { return hand; }
	
	std::string playerName();
	bool follow(CardSet &, CardSet &);
	bool approve(CardSet & );
	void cardSetOfSameRanks(CardSet &, int);	
	
};

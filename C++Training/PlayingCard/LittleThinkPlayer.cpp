/*
 *  Player.cpp
 *  PlayingCard
 *
 *  Created by 下薗 真一 on 09/04/12.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include <string>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"
#include "LittleThinkPlayer.h"

LittleThinkPlayer::LittleThinkPlayer(char * s) : Player(s) {}

bool LittleThinkPlayer::approve(CardSet & pile) {
	return true;
}

bool LittleThinkPlayer::follow(CardSet & pile, CardSet & s) {
	Card tmp;
	s.makeempty();
	inHand().pickup(&tmp, -1); // anyway, choose a card.
	s.insert(tmp);
	// the card idential to tmp is already removed from the hand. 
	cardSetOfSameRanks(s, pile.size());		
	 // this makes no card set with no more than 4 cards and not including Jkr.
	return true;
}



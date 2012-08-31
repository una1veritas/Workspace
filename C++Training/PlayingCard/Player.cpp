/*
 *  Player.cpp
 *  PlayingCard
 *
 *  Created by on 09/04/12.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
#include <stdio.h>
#include <string.h>

#include <iostream>
#include <string>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"


Player::Player(char * given) {
	hand.makeempty();
	name = given;  // constructor of std::string casts char array to std::string
	return;
}

Player::Player(std::string given) {
	hand.makeempty();
	name = given;
	return;
}


std::string Player::playerName() { 
	return name; 
}


std::string Player::printString() const {
	std::string tmp;
	tmp = name + ": ";
	tmp += hand.printString();
	return tmp;
}

void Player::clearHand() {
	hand.makeempty();
}

bool Player::isEmptyHanded() {
	return hand.isempty();
}

bool Player::pickup(Card c) {
	return hand.insert(c);
}

bool Player::takeCards(CardSet & s) {
	Card tmp;
	while (!s.isempty()) {
		s.pickup(&tmp, 0);
		hand.insert(tmp);
	}
	return true;
}

bool Player::approve(CardSet & pile) {
	return true;
}

bool Player::follow(CardSet & pile, CardSet & s) {
	Card tmp;
	s.makeempty();
	hand.pickup(&tmp, -1); // anyway, choose a card.
	s.insert(tmp);
	return true;
}

void Player::cardSetOfSameRanks(CardSet & cs, int pileSize) {
	Card c;
	for (int i = 0; i < hand.size() && (pileSize == 0 || cs.size() < pileSize)
					&& (cs.size() < 5); ) {
		if (hand[i].getrank() != cs[0].getrank()) {
			i++;
			continue;
		}
		hand.pickup(& c, i);
		cs.insert(c);
	}
}

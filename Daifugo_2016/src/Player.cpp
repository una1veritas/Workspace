/*
 *  Player.cpp
 *  PlayingCard
 *
 *  Created by 下薗 真一 on 09/04/12.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
#include <stdio.h>
#include <string.h>

#include <iostream>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"

#include "GameStatus.h"


bool Player::cardGreaterThan(const Card & one, const Card & another) {
	// assuming Joker is unique
	if ( one.isJoker() )
		return true;
	if ( another.isJoker() )
		return false;
	return (one.getNumber() + 10) % 13 > (another.getNumber() + 10) % 13;
}

Player::Player(char const * given) {
	hand.makeEmpty(); //clear();
	id = 0;
	name = given; // convert to std::string
	return;
}

std::ostream & Player::printOn(std::ostream & out) const {
	out << name << ": ";
	hand.printOn(out);
	return out;
}

// depricated.
/*
void Player::setId(int temp){
  id=temp;
}

int Player::getId() const{
  return id;
}
*/

void Player::clearHand() {
	hand.makeEmpty();
}

bool Player::isEmptyHanded() const {
	return hand.isEmpty();
}

bool Player::insert(Card c) {
	return hand.insert(c);
}

bool Player::takeCards(CardSet & s) {
	if (!s.isEmpty())
		hand.insertAll(s);
	return true;
}

void Player::ready(void) {

}

bool Player::approve(const GameStatus & gstat) {
  return true;
}

bool Player::follow(const GameStatus & gstat, CardSet & cards) {
	CardSet pile(gstat.pile);
	Card tmp;
	/*
	 *  cards はカラの状態で渡される．
	 *  ランダムに手札から一枚選ぶ．
	 */
	hand.pickup(tmp, -1);
	/*
	 * いったん tmp のカードは手札からなくなる．
	 * reject された場合は，勝手に手札にもどされる．
	 */
	cards.insert(tmp);
	return true;
}


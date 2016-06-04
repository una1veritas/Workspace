/*
 *  NonakaPlayer.cpp
 *  PlayingCard
 *
 *  Created by Gaku Nonaka on 16/05/19.
 *
 *  Created by 下薗 真一 on 09/04/12.
 *  Modified by Kazutaka Shimada on 09/04/21.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include <string>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"

#include "Group5.h"

void Group5::ready() {
	// 最初にカードを配られた状態
	mymemory.makeEmpty(); //memory.clear();
	sort();
}

bool Group5::approve(const GameStatus & gstat){
	CardSet pile(gstat.pile);
	mymemory.insert(pile);
	return true;
}

bool Group5::follow(const GameStatus & gstat, CardSet & s) {
	CardSet pile(gstat.pile);
	CardSet mycards;
	Card card, tmp;
//	std::cout << std::endl << gstat << std::flush;
	sort();
//	std::cout << "average: " << average(hand) << std::endl;
//	std::cout << inHand() << std::endl;
	int num, num1, a, b, c, d, i, j;
	sort();
	if(pile.isEmpty()){
	  for(i = 0; i < hand.size() && hand[i].getNumber() <= 10 && hand[i].getNumber() > 2; i++) {
			num = same(i);
			if(num >= 2) {
				for(j = i;j < i + num;j++) {
					hand.pickup(tmp,i);
					s.insert(tmp);
				}
				return true;
			}
		}
		num1 = hand[0].getNumber();
		inHand().pickup(tmp,0);
		s.insert(tmp);
		while( !hand.isEmpty() && num1 == hand[0].getNumber()) {
			hand.pickup(tmp, 0);
			s.insert(tmp);
		}
		return true;
	}
	a = 1;
	d = mymemory.size();
	for(i = 1; i < (4 <= pile.size() ? 4 : pile.size()) ; i++){
		if(pile[0].getNumber( ) == pile[i].getNumber()) {
			a++;
		}
	}
	b = pile[0].getNumber();
	if(b == 1 || b == 2){
		b = b + 13;
	}

	for(i = 0; i < hand.size(); i++) {
		c = hand[i].getNumber();
		num = same(i);
		if(c != 0){
			if(c == 1 || c == 2) {
				c = c + 13;
			}
			if(d < 20) {	// 20枚以下の時
				if(c < 14) {
					if(num == a && c > b) {
						for(j = i;j < i + a;j++) {
							hand.pickup(tmp,i);
							s.insert(tmp);
						}
						return true;
					}
				}
			} else {	// 20枚以上の時
				if(num == a && a == 1 && c > b) {
				//if((num == a || hand.size() == num) && a == 1 && c > b) {
				//if((num == a || hand[num].getNumber == NULL) && a == 1 && c > b) {
					inHand().pickup(tmp,i);
					s.insert(tmp);
					return true;
				}
				if(num >= a && c > b) {
					for(j = i; j < i + a; j++) {
						inHand().pickup(tmp,i);
						s.insert(tmp);
					}
					return true;
				}
			}
		} else {
			if(b == 15 && a == 1 && d < 20){
				inHand().pickup(tmp,i);
				s.insert(tmp);
				return true;
			} else if(a == 1 && d > 20) {
				inHand().pickup(tmp,i);
				s.insert(tmp);
				return true;
			}
		}
		i = i + num - 1;
	}
	// the card identical to tmp is already removed from the hand.
	// cardSetOfSameRanks(s, pile.size());
	// たとえば、複数枚のカードを探す関数。ただしこの関数は未実装。
	// 現状ではこの follow は Player.cpp のものと等
	return true;
}

int Group5::same(int a) {
	int i, count, num;
	count = 1;
	num = hand[a].getNumber();
	for(i = a; i+1 < hand.size() && num == hand[i + 1].getNumber(); i++) {
		if(i < hand.size() - 1) {
			count++;
		}
	}
	return count;
}

/*
 * 順序関係 compareCards を使うナイーヴ naive なソート．
 * 枚数は少ないので，効率は気にしない．
 */
void Group5::sort(bool ascending) {
	for(int i = 0; i + 1 < hand.size(); i++) {
		for(int j = i + 1; j < hand.size(); j++) {
			if ( (ascending && cardGreaterThan(hand[i], hand[j]))
					|| (!ascending && cardLessThan(hand[i], hand[j])) ) {
				Card t = hand[i];
				hand[i] = hand[j];
				hand[j] = t;
			}
		}
	}
}

//   Returns true if and only if the left CardSet is either correct and stronger than
// the right one, or the right one is an illegal set.
//
bool Group5::cardsStrongerThan(const CardSet & left, const CardSet & right) {
	int leftRank, rightRank;

	// regarded as "pass"
	if (left.isEmpty() )
		return false;

	// left is an illegal set
	if (!checkRankUniqueness(left))
		return false;
	if ( left.size() >= 5 )
		return true;

	// left always wins
	if ( right.isEmpty() )
		return true;

	// right is an illegal set.
	if (!checkRankUniqueness(right))
		return true;
	if ( right.size() >= 5 )
		return true;

	// the number of cards of the left set must be match to that of the right one.
	if ( left.size() != right.size() )
		return false;


	leftRank = cardStrength(left);
	rightRank = cardStrength(right);

	if ( leftRank > rightRank )
		return true;
	else
		return false;
}

int Group5::cardStrength(const CardSet & cs) {
	int i;
	if ( cs.isEmpty() )
		return 0;

	if ( cs.size() == 1 && cs[0].isJoker() ) {
		return cardStrength(cs[0]);
	}
  	for (i = 0; i < cs.size(); i++) {
	  if (!cs[i].isJoker()) {
		  break;
	  }
	}
	return cardStrength(cs[i]);
}

bool Group5::checkRankUniqueness(const CardSet & cs) {
	int rank = 0;

	if (cs.size() == 0)
		return false;

	if ( cs.size() == 1 && cs[0].isJoker() )
		return true;

	for (int i = 0; i < cs.size(); i++) {
	  if (cs[i].isJoker() )
		  continue;  // Jkrをスキップ
	  if ( rank == 0 ) {
		  rank = cs[i].getNumber();
	  } else if ( rank != cs[i].getNumber() ) {
	    return false;
	  }
	}
	return true;
}

double Group5::average(const CardSet & cset) {
	double sum = 0;
	if ( cset.size() == 0 )
		return sum;
	for(int i = 0; i < cset.size(); i++) {
		sum += cardStrength(cset[i]);
	}
	return sum/cset.size();
}

int Group5::cardStrength(const Card & c) {
	if ( c.isJoker() )
		return 18;
  	if ( c.getNumber() <= 2 )
  		return c.getNumber() + 13;
	return c.getNumber();
}

CardSet & Group5::findSmallestAcceptable(const CardSet & cs, CardSet & mycs) {
	// assumes the hand is sorted in ascending order
	mycs.makeEmpty();
	int cssize = cs.size();
	int csstrength = cardStrength(cs);
	int i, n;
	if ( cssize > 1 )
		std::cout << "multiple cards!!!" << std::endl;
	for(i = 0; i < hand.size(); ) {
		if ( cardStrength(hand[i]) <= csstrength ) {
			++i;
			continue;
		}
		for(n = 1; i + n < hand.size(); ) {
			if ( hand[i+n].isJoker()
				|| (cardStrength(hand[i + n]) == cardStrength(hand[i])) ) {
				++n;
				continue;
			}
			break;
		}
		if ( cssize <= n ) {
			if ( cssize != 0 )
				n = cssize;
			for(int j = i; j < i + n; j++)
				mycs.insert(hand[j]);
			return mycs;
		}
		i += n;
	}
	return mycs;   // empty card set.
}

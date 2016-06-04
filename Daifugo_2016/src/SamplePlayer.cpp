/*
 *  SamplePlayer.cpp
 *  PlayingCard
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

#include "SamplePlayer.h"

void SamplePlayer::ready() {
	// 最初にカードを配られた状態
	mymemory.makeEmpty(); //memory.clear();
	sort();
}

bool SamplePlayer::follow(const GameStatus & gstat, CardSet & s) {
	CardSet pile(gstat.pile);
	CardSet mycards, nextmycards;
	Card card;
	double average;

	//std::cout << gstat << std::flush;
	sort();
	average = totalStrength(inHand()) / inHand().size() ;
	std::cout << "average: " << average << " " << std::flush;
	//std::cout << inHand() << std::endl;

	if ( cardStrength(pile) < average ) {
		findSmallestAcceptable(pile, mycards);
		if ( !mycards.isEmpty() ) {
			inHand().removeAll(mycards);
			s.insertAll(mycards);
		} // else leaves s as empty card set.
		return true;
	} else {
		std::cout << "Hummm..." << std::flush;
		findSmallestAcceptable(pile, mycards);
		while ( !mycards.isEmpty() ) {
			if ( (std::rand()/ ((double)RAND_MAX)) > 0.5 ) {
				inHand().removeAll(mycards);
				s.insertAll(mycards);
				return true;
			}
			nextmycards.clear();
			findSmallestAcceptable(mycards, nextmycards);
			mycards = nextmycards;
		}
	}
	s.clear();
	// やっぱやめる
	return true;
}

/*
 * 順序関係 compareCards を使うナイーヴ naive なソート．
 * 枚数は少ないので，効率は気にしない．
 */
void SamplePlayer::sort(bool ascending) {
	for(int i = 0; i+1 < hand.size(); i++) {
		for(int j = i+1; j < hand.size(); j++) {
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
bool SamplePlayer::cardsStrongerThan(const CardSet & left, const CardSet & right) {
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

int SamplePlayer::cardStrength(const CardSet & cs) {
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

bool SamplePlayer::checkRankUniqueness(const CardSet & cs) {
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

double SamplePlayer::totalStrength(const CardSet & cset) {
	double sum = 0;
	if ( cset.size() == 0 )
		return sum;
	for(int i = 0; i < cset.size(); i++) {
		sum += cardStrength(cset[i]);
	}
	return sum;
}

int SamplePlayer::cardStrength(const Card & c) {
	if ( !c.isValid() )
		return 0;
	if ( c.isJoker() )
		return 18;
  	if ( c.getNumber() <= 2 )
  		return c.getNumber() + 13;
	return c.getNumber();
}

CardSet & SamplePlayer::findSmallestAcceptable(const CardSet & pile, CardSet & result) {
	// assumes the hand is sorted in ascending order

	Card JOKER;
	JOKER.set(Card::SUIT_JOKER, 0);

	int pilesize = pile.size();
	int pilestrength = cardStrength(pile);
	int count;
	bool hasJoker;

	result.clear();
	hasJoker = hand[hand.size() - 1].isJoker();
	for(int i = 0; i < hand.size(); ++i) {
		if ( !(cardStrength(hand[i]) > pilestrength) )
			continue;

		for(count = 1;
				(i + count < hand.size()) && (hand[i].getRank() == hand[i+count].getRank()) ;
				count++) ;
		if ( pilesize == 1 ) {
			result.insert(hand[i]);
			return result;
		} else if ( pilesize == 0 ) {
			for(int j = 0; j < count; j++)
				result.insert(hand[i+j]);
			return result;
		} else if ( pilesize <= count ) {
			for(int j = 0; j < pilesize; j++)
				result.insert(hand[i+j]);
			return result;
		} else if ( hasJoker && (pilesize == count + 1) ) {
			for(int j = 0; j < count; j++)
				result.insert(hand[i+j]);
			result.insert(hand[hand.size() - 1]);
			return result;
		}
	}
	return result;   // empty card set.
}

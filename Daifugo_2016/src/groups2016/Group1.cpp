/*
 *  Group1.cpp
 *  PlayingCard
 *
 *
 */
#include <iostream>
#include <string>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"

#include "Group1.h"

void Group1::ready() {
	// 最初にカードを配られた状態
	sort();
	pre = 20;
	passcount = 0;
	for(int i = 0; i < 16; i++)
		count[i] = 0;
	
}

bool Group1::approve(const GameStatus & gstat){
	CardSet pile(gstat.pile);
	int i;
	if(pre != cardStrength(pile)){
		for(i = 0; i < pile.size(); i++){
			count[cardStrength(pile[i]) - 3]+=1;
			//std::cout << cardStrength(pile[i])-3 << "count!" << std::endl;
		}
		pre = cardStrength(pile);
		//std::cout << "update" << pre << std::endl;
		passcount = 0;
	}else{
		passcount++;
		if(passcount == gstat.numPlayers-1){	//パスが続いて、親が変わった
			passcount = 0;
			pre = 20;
			//std::cout << "change parent" << std::endl;
		}
	}
	return true;
}

bool Group1::follow(const GameStatus & gstat, CardSet & s) {
	CardSet pile(gstat.pile);
	CardSet mycards;
	Card card;
	int i;
	//std::cout << std::endl << gstat << std::flush;
	sort();
	//std::cout << "average: " << average(hand) << std::endl;
	//std::cout << inHand() << std::endl;
	//std::cout << "count: ";
	//for(i = 0; i < 16; i++)
	//	std::cout << count[i] ;
	//std::cout << std::endl;
	//std::cout << "isHaveStrengest :";
	/*if(isHaveStrengest())
		std::cout << "true" << std::endl;
	else
		std::cout << "false" << std::endl;
		*/
	
		find(pile, mycards);
		if ( mycards.size() > 0 ) {
			hand.removeAll(mycards);
			s.insertAll(mycards);
			for(i = 0; i < s.size(); i++){
				count[cardStrength(s[i]) - 3]+=1;
			//	std::cout << cardStrength(s[i])-3 << "count!" << std::endl;
			}
			pre = cardStrength(s);
			passcount = 0;
			return true;
		}
		passcount++;
		if(passcount == gstat.numPlayers-1){	//パスが続いて、親が変わった
			passcount = 0;
			pre = 20;
			//std::cout << "change parent" << std::endl;
		}
	return true;
}


/*
 * 順序関係 compareCards を使うナイーヴ naive なソート．
 * 枚数は少ないので，効率は気にしない．
 * 引数trueなら右の方が強い
 */
void Group1::sort(bool ascending) {
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
bool Group1::cardsStrongerThan(const CardSet & left, const CardSet & right) {
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

int Group1::cardStrength(const CardSet & cs) {
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

//正しい一組だったらtrue
//たとえば、2が3枚ならtrue,3が2枚ジョーカー1枚ならtrue
//1が1枚4が1枚ならfalse
bool Group1::checkRankUniqueness(const CardSet & cs) {
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

double Group1::average(const CardSet & cset) {
	double sum = 0;
	if ( cset.size() == 0 )
		return sum;
	for(int i = 0; i < cset.size(); i++) {
		sum += cardStrength(cset[i]);
	}
	return sum/cset.size();
}

int Group1::cardStrength(const Card & c) {
	if ( c.isJoker() )
		return 18;
  	if ( c.getNumber() <= 2 )
  		return c.getNumber() + 13;
	return c.getNumber();
}

CardSet & Group1::find(const CardSet & cs, CardSet & mycs) {
	// assumes the hand is sorted in ascending order
	mycs.makeEmpty();
	int cssize = cs.size();
	int csstrength = cardStrength(cs);
	int i, j, n;
	if ( cssize >= 1 ){
		for(i = 0; i < hand.size(); ) {
			if ( cardStrength(hand[i]) <= csstrength ) {
				++i;
				continue;
			}
			for(n = 1; i + n < hand.size(); ) {
				if (cardStrength(hand[i + n]) == cardStrength(hand[i]) ) {
					++n;
					continue;
				}
				break;
			}
			if ( cssize == n ) {
				for(j = i; j < i + n; j++)
					mycs.insert(hand[j]);
				return mycs;	//なるべく組で出せるカードは崩さない
			}
			i += n;
		}
		for(i = 0; i < hand.size(); ) {
			if ( cardStrength(hand[i]) <= csstrength ) {
				++i;
				continue;
			}
			for(n = 1; i + n < hand.size(); ) {
				if (cardStrength(hand[i + n]) == cardStrength(hand[i]) ) {
					++n;
					continue;
				}
				break;
			}
			if ( cssize <= n ) {
				n = cssize;
				for(j = i; j < i + n; j++)
					mycs.insert(hand[j]);
				return mycs;	//なかったら組を崩してでもカード出す
			}
			i += n;
		}
			
	}else{	//自分が親だったとき(cssize == 0?)なるべく組となるカードを出す
		for(i = 0; i < hand.size();){
			for(n = 1; i + n < hand.size();){
				if(cardStrength(hand[i+n]) == cardStrength(hand[i])){
					++n;
					continue;
				}
				break;
			}
			if(n > 1){
				if(cardStrength(hand[i]) > 9 && i > 0 && cardStrength(hand[0]) < 10){
					mycs.insert(hand[0]);
					return mycs;	//10以上のペアより、9以下シングルを優先して出す
				}
				for(j = i; j < i + n; j++)
					mycs.insert(hand[j]);
				return mycs;
			}
			i += n;
		}
		mycs.insert(hand[0]);
		return mycs;	//なかったら小さいのから出す
	}
	return mycs;   // empty card set.
}

bool Group1::isHaveStrengest(){
	int i = 15;
	if(count[i] == 1){
		for(i = 14; i > -1; i--){
			if(count[i] == 4)
				continue;
			else
				if(i == cardStrength(hand[hand.size()-1]))
					return true;
				else
					return false;
		}
		return false;
	}else
		return hand[hand.size()-1].isJoker();
}

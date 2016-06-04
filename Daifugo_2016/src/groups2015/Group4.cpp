/*
 *  Group4.cpp
 *  PlayingCard
 *
 *  Created by 田村　玲人 on 15/05/28.
 *  Modified by Kazutaka Shimada on 09/04/21.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
#include <groups2015/Group4.h>
#include <iostream>
#include <string>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"
using namespace grp2015;

void Group4::ready() {
	memory.makeEmpty(); //memory.clear();
	trump.makeEmpty(); //trump.clear();
}

bool Group4::follow(const GameStatus & gstat, CardSet & s) {
	CardSet pile(gstat.pile);
	Card tmp;
	s.makeEmpty(); //clear();
	sortInHand();
	//ここから作成
	countCard(count);
	int pileSize = pile.size();
	Card pileCard;
	pile.pickup(pileCard);
	int pileNum = getCountCard(pileCard);
	
	if(pileSize == 0){
		for(int i = 4; i > 0; i--){
			for(int j = 0; j < 14; j++){
				if(count[j] == i){
					int target = getCountTargets(j);
					for(int k = 0; k < count[j]; k++){
						Card p;
						hand.pickup(p, target);
						s.insert(p);
					}
					return true;
				}
			}
		}
	}else{
		for(int i = pileNum + 1; i < 14; i++){
			if((count[i] == pileSize) || (i > 10 && count[i] > pileSize)){
				int target = getCountTargets(i);
				for(int k = 0; k < pileSize; k++){
					Card p;
					hand.pickup(p, target);
					s.insert(p);
				}
				return true;
			}
		}
	}
	//ここまで作成
	return true;
}
int Group4::getCountCard(const Card & c){
	int ret;
	if(c.getSuit() == 4)	ret = 13;
	else{
		ret = c.getNumber() - 3;
		if(ret < 0)	ret += 13;
	}
	return ret;
}

void Group4::countCard(int c[14]){
	for(int i=0; i < 14; i++)	c[i] = 0;
	
	for(int j = 0; j < hand.size(); j++){
		c[getCountCard(hand[j])]++;
	}
}

int Group4::getCountTargets(int p){
	int i = 13, ret = 0;
	while(1){
		if(p == i)	break;
		ret += count[i];
		i--;			
	}
	return ret;			
}
/*
 * ソートに使う順序関数の例．
 * 自分のならべたい順序の定義を作成する．
 * これはカード同士の比較であり，カードの集合同士の比較ではない．
 */
bool Group4::cardLessThan(const Card & c1, const Card & c2) {
	if ( getCountCard(c1) < getCountCard(c2) )	return true;
	return false;
}

/*
 * 順序関数と整合性のある同値（等しい）関係．
 */
bool Group4::cardEquals(const Card & c1, const Card & c2) {
	return !cardLessThan(c1, c2) && !cardLessThan(c2, c1);
}

/*
 * 順序関係 compareCards を使うナイーヴな naive ソートの例．
 * 枚数は少ないので，効率は気にしない．
 */
void Group4::sortInHand(void) {
	for(int i = 0; i+1 < hand.size(); i++) {
		for(int j = i+1; j < hand.size(); j++) {
			if ( cardLessThan(hand[i], hand[j]) ) {
				Card t = hand[i];
				hand[i] = hand[j];
				hand[j] = t;
			}
		}
	}
}


/*
 *  Player.cpp
 *  PlayingCard
 *
 *  Created by 下薗 真一 on 09/04/12.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
#include <extra/ThinkTA1.h>
#include <iostream>
#include <string>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"

bool ThinkTA1::follow(const GameStatus & gstat, CardSet & s) {
	Card tmp;
	s.clear();
	//std::cerr << "#####################" << std::endl;
	Card now, min;
	CardSet sortset(inHand());
	sort(sortset); //sorted
	//**********場が０枚**************
	if (gstat.pile.size() == 0) {
		/*  for(int i=0; i < sortset.size(); i++){
		 for(int j=1; j < )
		 if(sortset[i+1].getRank()){

		 }
		 }*/

		//微妙
		//1/2までで３枚組があるか探す;あったら出す
		for (int i = 0; i < sortset.size() / 2; i++) {
			if ( (i+1 < sortset.size()) && (sortset[i].getRank() == sortset[i + 1].getRank()) ) {
				if ( (i+2 < sortset.size()) && (sortset[i + 1].getRank() == sortset[i + 2].getRank()) ) { //３枚組あり
					sortset.pickup(tmp, i);
					inHand().remove(tmp);
					s.insert(tmp);
					sortset.pickup(tmp, i);
					inHand().remove(tmp);
					s.insert(tmp);
					sortset.pickup(tmp, i);
					inHand().remove(tmp);
					s.insert(tmp);
					//  sleep(2);
					return true;
				}
				/*else if(sortset[sortset.size()-1].isJoker()){ //２枚組あり&&Jokerあり
				 sortset.pickup(&tmp, i);
				 isIsHand()().remove(tmp);
				 s.insert(tmp);
				 sortset.pickup(&tmp, i);
				 isIsHand()().remove(tmp);
				 s.insert(tmp);
				 sortset.pickup(&tmp, sortset.size()-1);
				 isIsHand()().remove(tmp);
				 s.insert(tmp);
				 return true
				 }*/
			}
		}

		//半分までで２枚組があるか探す;あったら２枚出す
		for (int i = 0; i < sortset.size() / 2; i++) {
			if (sortset[i].getRank() == sortset[i + 1].getRank()) {
				sortset.pickup(tmp, i);
				inHand().remove(tmp);
				s.insert(tmp);
				sortset.pickup(tmp, i);
				inHand().remove(tmp);
				s.insert(tmp);
				//  sleep(2);
				return true;
			}
		}

		//半分までで組がないなら最弱を１枚出す
		sortset.pickup(tmp, 0);
		inHand().remove(tmp);
		s.insert(tmp);
		//std::cerr << "################" << std::endl;
		return true;
	}
	now = gstat.pile[0]; //場のカードが判明
	//**********場が１枚**************
	if (gstat.pile.size() == 1) {
		//枚数が少なくなるまでは強いのはとっておく
		int limit = inHand().size();
		if (sortset.size() > 5)
			limit = limit - 1;
		/* else { //2枚になったら強いのを先行で出す
		 sortset.pickup(&tmp,sortset.size()-1);
		 isIsHand()().remove(tmp);
		 s.insert(tmp);
		 return true;
		 //	limit = isIsHand()().size();
		 }*/
		for (int i = 0; i < limit; i++) {
			if (cardGreaterThan(sortset[i], now)) { //場より大きい
				//組になっているものはスキップ
				if ( (i+1 < sortset.size()) && (sortset[i].getRank() == sortset[i + 1].getRank()) ) {
					i++;
					if ( (i+1 < sortset.size()) && (sortset[i].getRank() == sortset[i + 1].getRank()) )
						i++;
					continue;
				}
				sortset.pickup(tmp, i);
				inHand().remove(tmp);
				s.insert(tmp);
				return true;
			}
		}
		//組を出し惜しみしている場合
		for (int i = 0; i < limit; i++)
			if (cardGreaterThan(sortset[i], now)) { //場より大きい
				sortset.pickup(tmp, i);
				inHand().remove(tmp);
				s.insert(tmp);
				return true;
			}
	}
	//**********場が２枚**************
	else if (gstat.pile.size() == 2) {
		//半分までで出せるなら出す
		int limit = sortset.size() - (int) (sortset.size() / 3);
		//int limit = sortset.size();
		for (int i = 0; i < limit; i++) {
			if (cardGreaterThan(sortset[i], now)) {
				if ( (i+1 < sortset.size() ) && (sortset[i].getRank() == sortset[i + 1].getRank()) ) {
					sortset.pickup(tmp, i);
					inHand().remove(tmp);
					s.insert(tmp);
					sortset.pickup(tmp, i);
					inHand().remove(tmp);
					s.insert(tmp);
					//  sleep(2);
					return true;
				}
				//Jokerがあったら&&エース以上が必要
				int over = 13;
				if (sortset[sortset.size() - 1].isJoker())
					if ((sortset[i].getRank() + 10) % 13 > (over + 10) % 13) {
						sortset.pickup(tmp, i);
						inHand().remove(tmp);
						s.insert(tmp);
						sortset.pickup(tmp, sortset.size() - 1);
						inHand().remove(tmp);
						s.insert(tmp);
						return true;
					}
			}
		}
	}
	//**********場が３枚**************
	else if (gstat.pile.size() == 3) {
		//半分までで３枚組があるか探す;あったら３枚出す
		for (int i = 0; i < sortset.size() / 2; i++) {
			if (cardGreaterThan(sortset[i], now)) { //２枚組あり
				if ( (i+1 < sortset.size() ) && (sortset[i].getRank() == sortset[i + 1].getRank()) ) {
					if ( (i+2 < sortset.size() ) && (sortset[i + 1].getRank() == sortset[i + 2].getRank()) ) { //３枚組あり
						sortset.pickup(tmp, i);
						inHand().remove(tmp);
						s.insert(tmp);
						sortset.pickup(tmp, i);
						inHand().remove(tmp);
						s.insert(tmp);
						sortset.pickup(tmp, i);
						inHand().remove(tmp);
						s.insert(tmp);
						//  sleep(2);
						return true;
					} else if (sortset[sortset.size() - 1].isJoker()) { //２枚組あり&&Jokerあり
						int over = 9;
						if ((sortset[i].getRank() + 10) % 13
								> (over + 10) % 13) {
							//sortset.pickup(&tmp, sortset.size()-1);
							//isIsHand()().remove(tmp);
							//s.insert(tmp);
							sortset.pickup(tmp, i);
							inHand().remove(tmp);
							s.insert(tmp);
							sortset.pickup(tmp, i);
							inHand().remove(tmp);
							s.insert(tmp);
							sortset.pickup(tmp, sortset.size() - 1);
							inHand().remove(tmp);
							s.insert(tmp);
							return true;
						}
					}
				}
			}
		}
	}
	return true;
}

bool ThinkTA1::sort(CardSet & set) {
	Card temp;
	for (int i = 0; i < set.size(); i++) {
		for (int j = i + 1; j < set.size(); j++) {
			if (cardGreaterThan(set[i], set[j])) {
				temp = set[i];
				set[i] = set[j];
				set[j] = temp;
			}
		}
	}

	return true;
}

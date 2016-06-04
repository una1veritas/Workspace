/*
 *  Player.h
 *  PlayingCard
 *
 *  Created by 松元 拓也 on 13/5/2~.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include "Card.h"
#include "CardSet.h"
#include "Player.h"
#include "GameStatus.h"

namespace grp2013 {

class MyCard {
private:
	CardSet cards;//1組のカード
	bool waiting;//使うのを待つか
	
public:
	MyCard(void){reset();}
	bool set(Card c);	//カードの追加
	
	void setWait(bool tf);	//使うか待つか
	bool isWait();		//使うのを待つ
	void reset();		//初期化
	
	bool equal(MyCard tgt);
	
	int getrank();		//カードの数値を得る
	bool isJoker();		//このカードはジョーカー
	
	CardSet getCards(){ return cards; }
	int getLevel(){ return cards.size(); } //このカードは何枚組か
};

class MyCardSet {
public:
//3,3,5,6,7,J,J なら5という数え方
	const static int maxnumcard = 53;	
private:
	int numcard;			// 現在の集合内のカード組数
	MyCard cards[maxnumcard];	// カードの組データ
private:
	int locate(MyCard target);
public:
	MyCardSet(void){ reset();}
	int getSetSize() { return numcard; }	//何組あるか
	MyCard at(int i);
	void atSetWait(int i,bool tf);
        bool set(CardSet cardset);
	void reset();		//初期化
	bool insert(MyCard card);
	bool remove(MyCard target);
	void print();
	int getWaitMyCardNum();

};

class GroupA : public Player {
public:
	CardSet memory;
	CardSet strongestCard;
	MyCardSet mycards; 
	MyCardSet remains;

public:
	GroupA(const char *);

	bool follow(const CardSet &, CardSet &);
	bool approve(const CardSet &, int[]);
	
	bool follow(const GameStatus & gstat, CardSet & cards) {
	  return follow(gstat.pile, cards);
	}
	bool approve(const GameStatus & gstat) {
	  return approve(gstat.pile, (int *) gstat.numCards);
	}

	bool sort(CardSet &);
	void calcRemains(CardSet addRemains);
	void calcRemains();
	CardSet strongestNow(int num);
	bool thinkAndSetCard(const CardSet &,CardSet &,CardSet &);
};

}


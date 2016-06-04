/*
 *  Player.h
 *  PlayingCard
 *
 *  Created by 下薗 真一 on 09/04/12.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef _PLAYER_H_
#define _PLAYER_H_

#include "GameStatus.h"

/*
 * プレイヤークラスの基底クラス．
 * Player を public の基底クラスとし，グループのプレイヤーを拡張クラスとして
 * 実装する．
 * 他のグループのプレイヤークラス，大会プログラムと互換性がなくなりコンパイルできなく
 * なる可能性があるので，Player クラスの定義とソースコードは，変更しないこと．
 */
class Player {
	friend class Dealer;
private:
	unsigned long id;
	std::string name;

protected:
	CardSet hand;

public:
	bool takeCards(CardSet &);

public:
	/*
	 * ユーティリティ関数
	 * 大富豪ゲームでの手札の強さにもとづく同値関係と順序関係
	 *
	 */
	static bool cardGreaterThan(const Card & one, const Card & another);
	static bool cardLessThan(const Card & one, const Card & another) {
		return cardGreaterThan(another, one);
	}
	static bool cardEquals(const Card & one, const Card & another) {
		return !cardGreaterThan(one, another) && !cardGreaterThan(another, one);
	}

public:
	// 拡張した派生クラスでオーバーライドする関数
	Player(const char * myname);
    virtual ~Player() { }

	virtual void ready(void);
	virtual bool follow(const GameStatus & gstat, CardSet & cards);
	virtual bool approve(const GameStatus & gstat);

	void setID(int i) { id = i; }
	int getID() const { return id; }
	void clearHand();
	bool isEmptyHanded() const;
	bool insert(Card );
	CardSet & inHand() { return hand; }
	int size() const { return hand.size(); }
	std::string playerName() const { return name; }

	std::ostream & printOn(std::ostream & out) const;

};

#endif

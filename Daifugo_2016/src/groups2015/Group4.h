/*
 *  Group4.h
 *  PlayingCard
 *
 *  Created by 田村　玲人 on 15/05/28.
 *  Copyright 2015 group4. All rights reserved.
 *
 */

#include "Player.h"

namespace grp2015 {

class Group4 : public Player {
  CardSet memory;
  CardSet trump;
  int count[14];

public:
  Group4(const char * name = "Group4") : Player(name) {
  }

  /*
   * グループで実態を作成し思考処理を追加する関数．
   */
  // ゲームを始めるにあたり必要な初期化をする．
  virtual void ready();
  // カードを出す思考処理を組み込む．
  virtual bool follow(const GameStatus &, CardSet &);

  /*
   * 思考処理を実装するのに使うユーティリティ関数は，自由につくってよい．
 　* たとえば手札のソート，ソート順のもとで同等のカードの判別を行うための
   * 順序比較と同値検証の関数，ソート関数など．
   */
	bool cardLessThan(const Card & c1, const Card & c2);
	bool cardEquals(const Card & c1, const Card & c2);
	void sortInHand();
	
	int getCountCard(const Card & c);
	void countCard(int [14]);
	int getCountTargets(int p);

};

}

/*
 *  Group1.h
 *  PlayingCard
 *
 *  Created by 下薗 真一 on 09/04/12.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include "Player.h"

namespace grp2015 {

class Group1 : public Player {
  CardSet memory;
  CardSet trump;

public:
  Group1(const char * name = "Little John") : Player(name) {
	  /*
	   * 必要ならば初期化を書く．
	   * 基底クラス Player の初期化の継承部分（ : の後の Player(name)　の部分）
	   * は変更しない．name = の後の部分は，デフォルト（引数を省略した場合の標準引数値）
	   * として，他のグループと違う名前をつけるとよい．
	   */
  }

  /*
   * グループで実態を作成し思考処理を追加する関数．
   */
  // ゲームを始めるにあたり必要な初期化をする．
  virtual void ready();
  // カードを出す思考処理を組み込む．
  virtual bool follow(const GameStatus &, CardSet &);
  // プレイヤーの手札数の出力、CardSet memoryに場札に出現したカードを記憶
  virtual bool approve(const GameStatus & gstat);
  /*
   * 思考処理を実装するのに使うユーティリティ関数は，自由につくってよい．
 　* たとえば手札のソート，ソート順のもとで同等のカードの判別を行うための
   * 順序比較と同値検証の関数，ソート関数など．
   */
	bool cardLessThan(const Card & c1, const Card & c2);
	bool cardEquals(const Card & c1, const Card & c2);
	void sortInHand();
	int  numofSameCard(const Card & card);

};

}  // namespace


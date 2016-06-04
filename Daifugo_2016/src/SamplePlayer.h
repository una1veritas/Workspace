/*
 *  SamplePlayer.h
 *  PlayingCard
 *
 *  Created by 下薗 真一 on 09/04/12.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include <cstdlib>
#include <ctime>

#include "Player.h"

/*
 * あまり考えないプレーヤー．
 * グループでプレーヤークラスを拡張する際の参考のため．
 */
class SamplePlayer : public Player {
  CardSet mymemory;

public:
  SamplePlayer(const char * name = "Sample Tom") : Player(name) {
	  /*
	   * 必要ならば初期化を書く．
	   * 基底クラス Player の初期化の継承部分（ : の後の Player(name)　の部分）
	   * は変更しない．name = の後の部分は，デフォルト（引数を省略した場合の標準引数値）
	   * として，他のグループと違う名前をつけるとよい．
	   */
	  std::srand(std::time(0));
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
   * たとえば手札のソート順のもとでのソート．
   * カード一枚どうしの比較は Player クラスから継承
   */
	void sort(bool ascending = true);
	// CardSet 比較用ツール．基本 Dealer の使っている関数とおなじ．
	static bool cardsStrongerThan(const CardSet & left, const CardSet & right);
	static bool checkRankUniqueness(const CardSet & cs);
	static int cardStrength(const Card & c);
	static int cardStrength(const CardSet & cs);

	double totalStrength(const CardSet & );
	CardSet & findSmallestAcceptable(const CardSet & cs, CardSet & result);
};

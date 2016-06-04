/*
 *  Group1.h
 *  PlayingCard
 *
 *
 */
#ifndef Group1_H
#define Group1_H

#include "Player.h"

/*
 */
class Group1 : public Player {
  int count[16];	//cardStrength-3を添字として与え、カウントする
  int pre;	//カウントで使用
  int passcount;	//パスが何回続いているか

public:
  Group1(const char * name = "Group1") : Player(name) {
	  /*
	   * 必要ならば初期化を書く．
	   * 基底クラス Player の初期化の継承部分（ : の後の Player(name)　の部分）
	   * は変更しない．name = の後の部分は，デフォルト（引数を省略した場合の標準引数値）
	   * として，他のグループと違う名前をつけるとよい．
	   */
  }


  // ゲームを始めるにあたり必要な初期化をする．
  void ready();
  // カードを出す思考処理を組み込む．
  bool follow(const GameStatus &, CardSet &);

  bool approve(const GameStatus & gstat);

  ~Group1(){}
  
	void sort(bool ascending = true);
	// CardSet 比較用ツール．基本 Dealer の使っている関数とおなじ．
	static bool cardsStrongerThan(const CardSet & left, const CardSet & right);
	static bool checkRankUniqueness(const CardSet & cs);
	static int cardStrength(const Card & c);
	static int cardStrength(const CardSet & cs);

	double average(const CardSet & );
	CardSet & find(const CardSet & cs, CardSet & mycs);
	bool isHaveStrengest();
};

#endif

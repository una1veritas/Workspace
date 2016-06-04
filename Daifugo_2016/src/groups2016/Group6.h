#include "Player.h"


class Group6 : public Player {
  CardSet mymemory;

public:
  int trashcards;
  CardSet prpile;
  Group6(const char * name = "Group6") : Player(name) {
  }

  /*
   * グループで実態を作成し思考処理を追加する関数．
   */
  // ゲームを始めるにあたり必要な初期化をする．
  virtual void ready();
  // カードを出す思考処理を組み込む．
  virtual bool follow(const GameStatus &, CardSet &);
  virtual bool approve(const GameStatus & gstat);
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

	double myaverage(const CardSet & );
	CardSet & findSmallestAcceptable(const GameStatus & gstate,const CardSet & cs, CardSet & mycs);
	CardSet & StrongestCards(CardSet & mycs);
	int mycardsize(CardSet mycs);
	double numcardAve(const GameStatus & gstate);
	int smallestNumCard(const GameStatus & gstate);
	int midnum(const CardSet & cards);
	int gettrashcards(void);

};


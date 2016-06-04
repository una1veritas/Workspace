
namespace grp2013 {

class GroupH : public Player {
  CardSet memory;


public:
	GroupH(const char * name = "Henry");

	bool follow(const CardSet &, CardSet &);
	bool approve(const CardSet &, int[]);

	bool follow(const GameStatus & gstat, CardSet & cards) {
		  return follow(gstat.pile, cards);
	  }
	  bool approve(const GameStatus & gstat) {
		  return approve(gstat.pile, (int *) gstat.numCards);
	  }

	void cardSetOfSameRanks(CardSet &,CardSet &, int); /* follow()の際に複数のカードを探す::未実装 */
	CardSet LargeRanks(CardSet &, CardSet &); // 手札から場に出ているカードより大きいカードを全て取り出す
	Card SmallestRank(CardSet &); // 自身から一番小さいカードを取り出す
	void sort(); //カードのソート
	int cardnum(); //それぞれのカードの数字ごとの枚数を管理

};

}

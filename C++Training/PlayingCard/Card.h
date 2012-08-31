//
// Card - トランプカード型
//

class Card {
	// メンバ変数
	private:
		int suit;	// 組
		int rank;	// 番号
		
	// メンバ関数
	public:
	// トランプの組(suit)のコード
	enum SUIT {
		SUIT_SPADE,
		SUIT_DIAMOND,
		SUIT_HEART,
		SUIT_CLUB,
		SUIT_JOKER
	};

	Card(void)	{ return; }
		// デフォルトコンストラクタ(初期値不定)

	void set(int s, int r)
		{ suit = s; rank = r; return; }
		// 自身に指定した組と番号を入れる

	bool equal(Card tgt)
		{ return suit == tgt.suit && rank == tgt.rank; }
		// 自身と tgt が同じか否かの判定 (true: 同; false: 異)

	int getrank(void)	{ return rank; }

	int getsuit(void)		{ return suit; }
		// アクセサ

	bool isJoker() { return suit == SUIT_JOKER; }
	bool isGreaterThan(Card c);

	bool scan(void);
		// 標準出力から自身に入力する(true: エラー; false: 正常終了)

	void print(void);
		// 自身の値を標準出力に出力する
	std::string printString() const;
};

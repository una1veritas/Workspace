//
// cardset.h - トランプカードの集合型(C++版)
//	作者: (あなたの名前); 日付: (完成した日付)
//
#ifndef CARDSET_H
#define CARDSET_H

// トランプの組(suit)のコード
enum {
	SUIT_SPADE,
	SUIT_DIAMOND,
	SUIT_HEART,
	SUIT_CLUB,
	SUIT_JOKER
};

//
// Card - トランプカード型
//
class Card {
// メンバ変数
private:
	int suit;	// 組
	int number;	// 番号
// メンバ関数
public:
	Card(void)	{ }
		// デフォルトコンストラクタ(初期値不定)
	void set(int st, int num)	{ suit = st; number = num; }
		// 自身に指定した組と番号を入れる
	bool equal(Card tgt)
		{ return suit == tgt.suit && number == tgt.number; }
		// 自身と tgt が同じか否かの判定 (true: 同; false: 異)
	int gnumber(void)	{ return number; }
	int gsuit(void)		{ return suit; }
		// アクセサ
	bool scan(void);
		// 標準出力から自身に入力する(true: エラー; false: 正常終了)
	void print(void);
		// 自身の値を標準出力に出力する
};

//
// CardSet - トランプカードの集合型
//
class CardSet {
// 定義・宣言
public:
	const static int maxnumcard = 53;	// カード総数
// メンバ変数
private:
	int numcard;		// 現在の集合内のカード数
	Card cdat[maxnumcard];	// カードのデータ
// メンバ関数
private:
	int locate(Card target);
		// 内部での target のカードの位置を返す(-1: ない)
	int locate(int num);
		// 内部での num 番のカードの位置を返す(-1: ない)
public:
	CardSet(void)		{ makeempty(); }
		// デフォルトコンストラクタ(初期値空集合)
	void makeempty(void)	{ numcard = 0 ; }
		// 自身を空集合にする
	bool isempty(void)	{ return numcard == 0; }
		// 自身が空集合か否かの判定 (true: 空; false: 非空)
	void makedeck(void);
		// 自身に全部の(maxnumcard 枚の)カードを入れる
	bool pickup(Card* ret, int targetpos = -1);
		// 自身から targetpos 枚目のカードを除き *ret に返す
		// targetpos が -1 のときはランダムに選ぶ
		// (true: 失敗; false: 成功)
	bool insert(Card newcard);
		// 自身に newcard のカードを入れる(true: 失敗; false: 成功)
	bool remove(Card target);
		// 自身から target のカードを除く(true: 失敗; false: 成功)
	bool remove(int num);
		// 自身から num 番のカードを除く(true: 失敗; false: 成功)
	void print(void);
		// 自身の状態を標準出力に出力する
		
	// Streaming
	//
	friend std::ostream& operator<<(std::ostream& out, const CardSet & c);
	
};

#endif

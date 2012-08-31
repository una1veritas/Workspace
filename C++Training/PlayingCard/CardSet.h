//
// cardset.h - トランプカードの集合型(C++版)
//	作者: (あなたの名前); 日付: (完成した日付)
//
#ifndef CARDSET_H
#define CARDSET_H
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
	int size() { return numcard; }
	
	Card at(int);
	Card operator[](int);
	
	void makedeck(void);
		// 自身に全部の(maxnumcard 枚の)カードを入れる
	bool pickup(Card* ret, int targetpos = -1);
		// 自身から targetpos 枚目のカードを除き *ret に返す
		// targetpos が -1 のときはランダムに選ぶ
		// (true: 失敗; false: 成功)
	bool insert(Card newcard);
		// 自身に newcard のカードを入れる(true: 失敗; false: 成功)
	bool insert(CardSet & cards);

	bool remove(Card target);
		// 自身から target のカードを除く(true: 失敗; false: 成功)
	bool remove(int num);
		// 自身から num 番のカードを除く(true: 失敗; false: 成功)
	void print(void);
		// 自身の状態を標準出力に出力する
	std::string printString() const;
	
	void shuffle(void);
	  
	// Streaming
	//
	friend std::ostream& operator<<(std::ostream& out, const CardSet & c) {
		out << c.printString();
		return out;
	}
};

#endif

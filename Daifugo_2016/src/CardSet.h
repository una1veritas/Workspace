//
// CardSet.h - トランプカードの集合型(C++版)
//	作者: (あなたの名前); 日付: (完成した日付)
//
#ifndef CARDSET_H
#define CARDSET_H

#include <cstdlib>
#include <ctime>

#include "Card.h"

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
	Card cards[maxnumcard];	// カードのデータ

	// 内部で使用するメンバ関数
private:
	// target に等しいカードの格納位置を返す( == -1: ない)
	int find(const Card & target) const;
	// 数字が num に等しいカードの格納位置を返す(== -1: ない)
	int find(const int num) const;

public:
	// デフォルトコンストラクタ(初期値空集合)
	CardSet(void);

	CardSet(const CardSet & cset);

	// 自身をカラにする
	void clear(void);
	inline void makeEmpty(void)	{ clear(); }

	// 空集合か否かの判定 (true: 空; false: 非空)
	bool isEmpty(void) const { return numcard == 0; }
	bool equals(const CardSet & another) const;

	bool includes(const Card & target) const { return find(target) != -1; }

	int size(void) const { return numcard; }
	const Card & at(const int) const;
	Card & at(const int);
	inline const Card & operator[](const int i) const { return at(i); }
	inline Card & operator[](const int i) { return at(i); } 	// [ ] 演算子の定義．配列のように添字でアクセスできる

	// １デッキすべての(maxnumcard 枚の)カードを入れる
	void setupDeck(void);
	// newcard を追加する ( false: 要素数に変化なし; true: 追加成功)
	bool insert(const Card & newcard);
	inline bool insert(const CardSet & cards) { return insertAll(cards); }
	bool insertAll(const CardSet & cards);

	// target と同一のカードを取り除く (false: 要素数に変化なし; true: みつけて削除が成功)
	bool remove(const Card & target);
	// 数字が num であるカードいずれか一枚を除く(false: 要素数に変化なし; true: 成功)
	bool remove(int num);
	bool removeAll(const CardSet & );

	// targetpos 枚目のカードを除き，そのカードを card で返す　(false: 要素数に変化なし，card の値は変更されない; true: 成功)
	// targetpos が -1 の場合，乱数 random() を使ってランダムに選んで返す．
	bool pickup(Card & card, int targetpos = -1); 	// 引数を省略した場合 targetpos = -1

	// シャフルする
	void shuffle(const time_t & seed);
	void shuffle(void) { time_t seed; time(&seed); shuffle(seed); }

	// 自身の状態を標準出力に出力する
	void print(void) const;

	// おまけ

	bool operator==(const CardSet & another) const {
		return equals(another);
	}
	// Streaming
	//
	std::ostream&  printOn(std::ostream & out) const;
	friend std::ostream& operator<<(std::ostream& out, const CardSet & cset) {
		return cset.printOn(out);
	}

};

#endif

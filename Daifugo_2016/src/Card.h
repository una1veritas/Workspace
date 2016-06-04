//
// Card.h - トランプカードの型
//	作者: (あなたの名前); 日付: (完成した日付)
//
#ifndef CARD_H
#define CARD_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
//
// Card - トランプカード型
//
class Card {
	// クラス定数
	// トランプの組(suit)のコード
public:
	enum {
		SUIT_SPADE, 	// = 0
		SUIT_DIAMOND,
		SUIT_HEART,
		SUIT_CLUB,
		SUIT_JOKER, 	// = 4
		SUIT_BLANK
	};

	static const Card Joker;
	static const Card Blank;

// メンバ変数
private:
//	インスタンスメンバ変数．Card 型データ（オブジェクト）がそれぞれ持っているデータ．
	int suit;	// 組
	int rank;	// 番号

//	static は，クラスメンバのこと．Card クラスの中で共通の定数として Card::suitnames で参照できる．
//  クラス変数（定数）．値の初期化は .cpp で main の外に書いて行う

	static const char * suitnames[];
	static const char * suitabbrevnames[];
	static const char * ranknames[];

// メンバ関数
public:
	// デフォルトコンストラクタ（初期値はブランクカード）
	Card(void) : suit(SUIT_BLANK), rank(0) { }
	Card(const int s, const int r) : suit(s), rank(r) {}
	Card(const Card & c ) : suit(c.suit), rank(c.rank) {}

	// 組と番号を設定する
	void set(int st, int num) {
		suit = st;
		rank = num; 
	}

	// 自身と tgt のカードの組，番号が等しいか判定 (true: 同; false: 異)
	// 同じオブジェクト（同じアドレスにあるデータ）かどうか，ではない．
	bool equals(Card tgt) const {
		return (suit == tgt.suit) && (rank == tgt.rank); 
	}

	bool isValid() const {
		if ( ((SUIT_SPADE <= suit) && (suit <= SUIT_CLUB)) 
			 && (1 <= rank && (rank <= 13)) )
			return true;
		else if (suit == SUIT_JOKER)
			return true;
		return false;
	}
	
	bool isJoker() const { return suit == SUIT_JOKER; }

	// アクセサ
	int getSuit(void)	const {
		return suit;
	}

	int getRank(void) const {
		return rank;
	}

	// for backward compatibility
	inline int getNumber(void) const { return getRank(); }

	// 標準出力から自身に入力する(true: 正常終了; false: 異常終了)
	bool scan(void);
	
	// 自身の値を標準出力に出力する
	void print(void) const;

	// おまけ
	std::ostream & printOn(std::ostream & ostr) const;
	friend std::ostream & operator<<(std::ostream& ostr, const Card & card) {
		return card.printOn(ostr);
	}
};
#endif

//
// CardSet.cpp - トランプカードの集合型(C++版)
//	作者: (あなたの名前); 日付: (完成した日付)
//

#include "Card.h"
#include "CardSet.h"


//
// CardSet::remove() - card を削除する (false: 要素数に変化なし; true: 追加成功)
//
bool CardSet::remove(const Card & newcard) {
	int position = find(newcard);
	if( position == -1 )
		return false;	// はじめからない
	// 上書きして前詰め
	numcard--;
	for(int p = position; p < numcard; p++) {
		cards[p] = cards[p+1];
	}
	return true;
}

bool CardSet::remove(int num) {
	int position = find(num);
	if( position == -1 )
		return false;	// はじめからない
	// 上書きして前詰め
	numcard--;
	for(int p = position; p < numcard; p++) {
		cards[p] = cards[p+1];
	}
	return true;
}


bool CardSet::removeAll(const CardSet & origset) {
	bool flag = false;
	for(int i = 0; i < origset.numcard; i++) {
		if ( remove(origset.cards[i]) )
			flag = true;
	}
	return flag;
}


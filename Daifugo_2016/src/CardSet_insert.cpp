//
// cardset.cc - トランプカードの集合型(C++版)
//	作者: (あなたの名前); 日付: (完成した日付)
//
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <iostream>

#include "Card.h"
#include "CardSet.h"


CardSet::CardSet(const CardSet & orig) {
	numcard = orig.numcard;		// 現在の集合内のカード数
	for(int i = 0; i < numcard; i++)
		cards[i] = orig.cards[i];	// カードのデータ
}

/*
//
// CardSet::insert() - 自身に newcard を入れる(常に成功，挿入／存在位置を返す)
//
bool CardSet::insert(const Card & newcard)
{
	int location = find(newcard);
	if( location >= 0)
		return location;	// 既にある
	// 最後に追加
	if ( numcard >= maxnumcard )
		return -1; // もうはいらないし，入れるカードはないはず
	location = numcard;
	cards[location] = newcard;
	numcard++;

  return location;
}
*/


bool CardSet::insertAll(const CardSet & origset) {
	bool flag = false;
	for(int i = 0; i < origset.numcard; i++) {
		if ( insert(origset.cards[i]) )
			flag = true;
	}
	return flag;
}

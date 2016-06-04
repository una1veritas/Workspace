//
// CardSet.cpp - トランプカードの集合型(C++版)
//	作者: (あなたの名前); 日付: (完成した日付)
//
#include "Card.h"
#include "CardSet.h"


//
// CardSet::pickup() - 自身から targetpos 枚目のカードをぬき card にセットし返す
//	targetpos が -1 のときはランダムに選ぶ．
// (false: 失敗; true: 成功)
//
// remove() を実装後，コメントを外して使えるようにせよ
bool CardSet::pickup(Card & card, int targetpos)
{
	if ( numcard == 0 )
		return false;
	if ( targetpos < 0 || targetpos >= numcard)
		targetpos = random() % numcard;
	card = cards[targetpos];
	return remove(card); // 成功するはず
}



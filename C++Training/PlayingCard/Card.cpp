//
// Card.cpp - トランプカード(C++版)
//	作者: (あなたの名前); 日付: (完成した日付)
//
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <string>

#include "Card.h"

//
// Card::scan() - 標準出力から自身に入力する(true: エラー; false: 正常終了)
//
bool Card::scan(void)
{
	char buf[BUFSIZ];
	char* suitname[] = { "spade", "diamond", "heart", "club" };
	char* suitabbrevname[] = { "S", "D", "H", "C" };
// 4組のいずれかなら番号も入力する
	if(scanf("%s", buf) < 1)
		return true;
	for(int s = SUIT_SPADE; s <= SUIT_CLUB; s++)
		if(!strcmp(buf, suitname[s]) || !strcmp(buf, suitabbrevname[s])) {
			suit = s;
			if(scanf("%d", &rank) < 1)
				return true;
			if(rank < 1 || rank > 13)
				return true;
			return false;
		}
// joker はそのまま(rank は 0 とする)
	if(!strcmp(buf, "joker")) {
		suit = SUIT_JOKER;
		rank = 0x0f;
		return false;
	}

	return true;	// エラー
}

bool Card::isGreaterThan(Card another) {
	if (suit == SUIT_JOKER)
		return true;
	if (another.suit == SUIT_JOKER)
		return false;
	return (rank + 10) % 13 > (another.rank + 10) % 13;
}

//
// Card::print() - 自身の値を標準出力に出力する
//
void Card::print(void)
{
	char* suitname[] = { "S", "D", "H", "C" };
	char* symbol[] = { " X", " A", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", "10", " J", " Q", " K" };

	if(suit < SUIT_JOKER)
		printf("[%s%s]", suitname[suit], symbol[rank]);
	else if(suit == SUIT_JOKER)
		printf("[Jkr]");
}

std::string Card::printString() const {
	char* suitname[] = { "S", "D", "H", "C" };
	char* symbol[] = { " X", " A", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", "10", " J", " Q", " K" };
	std::string tmp;
	tmp = "[";
	if(suit < SUIT_JOKER)
		tmp = tmp + suitname[suit] + symbol[rank];
	else if(suit == SUIT_JOKER)
		tmp += "Jkr";
	tmp += "]";
	return tmp;
}


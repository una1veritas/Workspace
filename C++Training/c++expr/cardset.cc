//
// cardset.cc - トランプカードの集合型(C++版)
//	作者: (あなたの名前); 日付: (完成した日付)
//
#include <stdlib.h>
#include <stdio.h>
#include "cardset.h"


//
// Card::scan() - 標準出力から自身に入力する(true: エラー; false: 正常終了)
//
bool Card::scan(void)
{
	char buf[BUFSIZ];
	char* suitname[] = { "spade", "diamond", "heart", "club" };
// 4組のいずれかなら番号も入力する
	if(scanf("%s", buf) < 1)
		return true;
	for(int s = SUIT_SPADE; s <= SUIT_CLUB; s++)
		if(!strcmp(buf, suitname[s])) {
			suit = s;
			if(scanf("%d", &number) < 1)
				return true;
			if(number < 1 || number > 13)
				return true;
			return false;
		}
// joker はそのまま(number は 0 とする)
	if(!strcmp(buf, "joker")) {
		suit = SUIT_JOKER;
		number = 0;
		return false;
	}

	return true;	// エラー
}

//
// Card::print() - 自身の値を標準出力に出力する
//
void Card::print(void)
{
	char* suitname[] = { "spade", "diamond", "heart", "club" };

	if(suit < SUIT_JOKER)
		printf("[%s %d]", suitname[suit], number);
	else if(suit == SUIT_JOKER)
		printf("[joker]");
}

//
// CardSet::locate() - 内部での target のカードの位置を返す(-1: ない)
//
int CardSet::locate(Card target)
{
	for(int i = 0; i < numcard; i++)
		if(target.equal(cdat[i]))
			return i;

	return -1;	// 見つからなかった
}

//
// CardSet::locate() - 内部での num 番のカードの位置を返す(-1: ない)
//
int CardSet::locate(int number)
{
	for(int i = 0; i < numcard; i++)
		if(number == cdat[i].gnumber())
			return i;

	return -1;	// 見つからなかった
}

//
// CardSet::makedeck() - 自身に全部の(maxnumcard 枚の)カードを入れる
//
void CardSet::makedeck(void)
{
	Card c;
	int suit, num;

	for(suit = SUIT_SPADE; suit <= SUIT_CLUB; suit++)
		for(num = 1; num <= 13; num++) {
			c.set(suit, num);
			insert(c);
		}
	c.set(SUIT_JOKER, 0);
	insert(c);
}

//
// CardSet::pickup() - 自身から targetpos 枚目のカードを除き *ret に返す
//	targetpos が -1 のときはランダムに選ぶ(true: 失敗; false: 成功)
//
bool CardSet::pickup(Card* ret, int targetpos /* = -1 */)
{
	if(numcard == 0)
		return true;
	if(targetpos < 0)
		targetpos = random() % numcard;
	else
		targetpos %= numcard;

	*ret = cdat[targetpos];
	remove(*ret);

	return false;
}

//
// CardSet::insert() - 自身に newcard のカードを入れる(true: 失敗; false: 成功)
//
bool CardSet::insert(Card newcard)
{
	if(locate(newcard) >= 0)
		return true;	// 既にある
// 最後に追加
	cdat[numcard] = newcard;
	numcard++;

	return false;
}

//
// CardSet::remove() - 自身から target のカードを除く(true: 失敗; false: 成功)
//
bool CardSet::remove(Card target)
{
	int pos;

// 除くカードの位置を求める
	if((pos = locate(target)) < 0)
		return true;	// target のカードは無い
// 1つずつ前に詰める
	for(int i = pos + 1; i < numcard; i++)
		cdat[i-1] = cdat[i];
	numcard--;

	return false;
}

//
// CardSet::remove() - 自身から num 番のカードを除く(true: 失敗; false: 成功)
//
bool CardSet::remove(int number)
{
	int pos;

// 除くカードの位置を求める
	if((pos = locate(number)) < 0)
		return true;	// num 番のカードは無い
// 1つずつ前に詰める
	for(int i = pos + 1; i < numcard; i++)
		cdat[i-1] = cdat[i];
	numcard--;

	return false;
}

//
// CardSet::print() - 自身の状態を標準出力に出力する
//
void CardSet::print(void)
{
	printf("((CARDSET))\n");
	if(numcard == 0) {
		printf("\tno card\n");
		return;
	}
	for(int i = 0; i < numcard; i++) {
		printf("\t");
		cdat[i].print();
		printf("\n");
	}
}

}

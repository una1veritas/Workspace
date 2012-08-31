//
// cardset.cc - トランプカードの集合型(C++版)
//	作者: (あなたの名前); 日付: (完成した日付)
//
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string>

#include "Card.h"
#include "CardSet.h"


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
		if(number == cdat[i].getrank())
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

	for(suit = c.SUIT_SPADE; suit <=c.SUIT_CLUB; suit++)
		for(num = 1; num <= 13; num++) {
			c.set(suit, num);
			insert(c);
		}
	c.set(c.SUIT_JOKER, 0);
	insert(c);
}


Card CardSet::at(int i) {
	return cdat[i];
}

Card CardSet::operator[](int i) {
	return at(i);
}


//
// CardSet::pickup() - 自身から targetpos 枚目のカードを除き *ret に返す
//	targetpos が -1 のときはランダムに選ぶ(true: 成功; false: 失敗)
//
bool CardSet::pickup(Card* ret, int targetpos /* = -1 */)
{
	if(numcard == 0)
		return false;
	if(targetpos < 0)
		targetpos = random() % numcard;
	else
		targetpos %= numcard;

	*ret = cdat[targetpos];
	remove(*ret);

	return true;
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

bool CardSet::insert(CardSet & cards) {
	for(int i = 0; i < cards.numcard; i++) {
		insert(cards.cdat[i]);
	}
	return true;
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
	//printf("((CARDSET))\n\t");
	if(numcard == 0) {
		printf("no cards");
		return;
	}
	for(int i = 0; i < numcard; i++) {
		//printf("\t");
		cdat[i].print();
		printf(", ");
	}
	printf("\n");
}

std::string CardSet::printString() const {
	std::string tmp;
	if(numcard == 0) {
		tmp += "(no card)";
		return tmp;
	}
	for(int i = 0; i < numcard; i++) {
		tmp += cdat[i].printString();
		tmp += " ";
	}
	return tmp;
}


  void CardSet::shuffle(void) {
    int t, i, j;
    Card swap;
    time_t seed;
    
    time(&seed);
    srandom(seed);
    for (t = 0; t < 100; t++) {
      i = random() % numcard;
      j = ((random() % (numcard-1)) + 1 + i) % numcard;
      swap = cdat[i];
      cdat[i] = cdat[j];
      cdat[j] = swap;
    }
  }

//
// CardSet.cpp - トランプカードの集合型(C++版)
//	作者: (あなたの名前); 日付: (完成した日付)
//
#include <iostream>
#include "Card.h"
#include "CardSet.h"


CardSet::CardSet(void) {
	clear();
}

void CardSet::clear(void) {
	numcard = 0;
}

bool CardSet::equals(const CardSet & another) const {
	int i;
	if ( size() != another.size())
		return false;
	for(i = 0; i < size(); i++) {
		if ( ! this->includes(another[i]) )
			return false;
	}
	return true;
}


//
// CardSet::find() - 内部での target のカードの位置を返す(-1: ない)
//
int CardSet::find(const Card & target) const
{
  for(int i = 0; i < numcard; i++)
    if ( target.equals(cards[i]) )
      return i;
  
  return -1;	// 見つからなかった
}

//
// CardSet::find() - 内部で数字が num のカードの位置を返す(-1: ない)
//
int CardSet::find(int number) const
{
  for(int i = 0; i < numcard; i++)
    if(number == cards[i].getNumber())
      return i;

  return -1;	// 見つからなかった
}


Card & CardSet::at(const int i) {
	if ( i >= numcard ) {
		std::cerr << "err: std::out_of_range CardSet::at()" << std::flush;
//throw std::out_of_range("CardSet::at()");
	}
	return cards[i];
}

const Card & CardSet::at(const int i) const {
	if ( i >= numcard ) {
		std::cerr << "err: std::out_of_range CardSet::at()" << std::flush;
//		throw std::out_of_range("CardSet::at() const");
	}
	return cards[i];
}


//
// CardSet::setupDeck() - 自身に全部の(maxnumcard 枚の)カードを入れる
//
void CardSet::setupDeck(void)
{
  Card c;
  int suit, num;
  
  for(suit = Card::SUIT_SPADE; suit <= Card::SUIT_CLUB; suit++)
    for(num = 1; num <= 13; num++) {
      c.set(suit, num);
      insert(c);
    }
  c.set(Card::SUIT_JOKER, 0);
  insert(c);
}

//
// CardSet::pickup() - 自身から targetpos 枚目のカードをぬき card にセットし返す
//	targetpos が -1 のときはランダムに選ぶ．
// (false: 失敗; true: 成功)
//
// remove() を実装後，コメントを外して使えるようにせよ
/*
bool CardSet::pickup(Card & card, int targetpos)
{
	if ( numcard == 0 )
		return false;
	if ( targetpos < 0 || targetpos >= numcard)
		targetpos = random() % numcard;
	card = cdat[targetpos];
	return remove(card); // 成功するはず
}
*/


//
// CardSet::insert() - newcard を入れる (false: 要素数に変化なし; true: 追加成功)
//
bool CardSet::insert(const Card & newcard) {
	int position = find(newcard);
	if( position != -1 )
		return false;	// 既にある
	// 最後に追加
	if ( numcard >= maxnumcard )
		return false; // もうはいらないし，入れられるカードはないはず
	position = numcard;
	cards[position] = newcard;
	numcard++;

  return true;
}


//
// CardSet::print() - 自身を標準出力に出力する
//
void CardSet::print(void) const
{
  printf("((CARDSET))\n");
  if(numcard == 0) {
    printf("\tno card\n");
    return;
  }
  for(int i = 0; i < numcard; i++) {
    printf("\t");
    cards[i].print();
    printf("\n");
  }
}


std::ostream&  CardSet::printOn(std::ostream & out) const {
	//out << "CardSet";
	out << "(";
	if(numcard == 0) {
		out << " ) ";
		return out;
	}
	for(int i = 0; i < numcard; i++) {
		out << cards[i];
		if ( (i + 1) < numcard ) out << ", ";
	}
	out << ") ";
	return out;
}

//
// cardset.cpp - トランプカードの集合型(C++版)
//	作者: (あなたの名前); 日付: (完成した日付)
//
#include "Card.h"

#include <iostream>

// クラス変数（定数）の初期化

const Card Card::Joker(Card::SUIT_JOKER, 0);
const Card Card::Blank(Card::SUIT_BLANK, 0);

const char * Card::suitnames[] = {
  		"spade",
  		"diamond",
  		"heart",
  		"club",
  		"joker",
		"_blank"
};

const char * Card::suitabbrevnames[] = {
  		"S",
  		"D",
  		"H",
  		"C",
  		"Jkr",
		"_blank"
};

const char * Card::ranknames[] = {
  		"?",
  		"A",
  		"2",
  		"3",
  		"4",
  		"5",
  		"6",
  		"7",
  		"8",
  		"9",
  		"10",
  		"J",
  		"Q",
  		"K",
  		"_",
};

//
// Card::scan() - 標準出力などから値を得る (true: 取得成功; false: 取得失敗)
//
bool Card::scan(void)
{
  char buf[64], suitbuf[64], numbuf[64];
  int s;

  suit = SUIT_BLANK;
  // 4組のいずれ？
  if ( fgets(buf, 64, stdin) == NULL )
	  return false;
  sscanf(buf, "%s %s", suitbuf, numbuf);
  for(s = SUIT_SPADE; s <= SUIT_CLUB; s++) {
	  if( tolower(suitbuf[0]) == Card::suitnames[s][0] ) { // 一文字目だけで判定
		  suit = s;
		  break;
	  }
  }
  if ( SUIT_SPADE <= s && s <= SUIT_CLUB ) {
	  // なら番号も得る
	  rank = atoi(numbuf);
	  if ( 0 < rank && rank <= 13 ) {
		  return true;
	  }  else {
		  if ( rank == 0 ) {
			  switch( tolower(numbuf[0]) ) {
			  case 'a':
				  rank = 1;
				  break;
			  case 'j':
				  rank = 11;
				  break;
			  case 'q':
				  rank = 12;
				  break;
			  case 'k':
				  rank = 13;
				  break;
			  default:
				  return false;
			  }
			  return true;
		  }
		  return false;
	  }
  } else if( tolower(buf[0]) == 'j' ) {
  // joker は number を 0 としておく．
    suit = SUIT_JOKER;
    rank = 0;
    return true;
  }
  
  return false;	// エラー
}

//
// Card::print() - 自身の値を標準出力に出力する
//
void Card::print(void) const
{
	fprintf(stdout, "[%s %s]", suitnames[suit], ranknames[rank]);
}

std::ostream & Card::printOn(std::ostream& ostr) const {
	ostr << '[' << suitabbrevnames[suit];
	if (suit != SUIT_JOKER )
		ostr << " " << ranknames[rank];
	ostr << ']';
	return ostr;
}


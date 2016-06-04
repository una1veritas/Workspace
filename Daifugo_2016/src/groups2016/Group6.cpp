#include <iostream>
#include <string>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"

#include "Group6.h"

void Group6::ready() {
	// 最初にカードを配られた状態
	mymemory.makeEmpty(); //memory.clear();
	trashcards = 0;
	sort();
}

bool Group6::follow(const GameStatus & gstat, CardSet & s) {
	CardSet pile(gstat.pile);
	CardSet mycards;
	Card card;
//	std::cout << std::endl << gstat << std::flush;
	sort();
//	std::cout << "average: " << myaverage(hand) << std::endl;
//	std::cout << inHand() << std::endl;
//	std::cout << "mynumcard:" << hand.size() <<std::endl;
//	std::cout << "numcardave:" << numcardAve(gstat) <<std::endl;
//	std::cout << "numcardmin:" << smallestNumCard(gstat) <<std::endl;
//	std::cout << "midmycard:" << midnum(hand) <<std::endl;
//
//
	findSmallestAcceptable(gstat,pile,mycards);
	
	if(cardStrength(pile) < cardStrength(mycards)){
	  if(((hand.size() >= 5) && (cardStrength(mycards) >= 14) && !(smallestNumCard(gstat) <= 5) && (cardStrength(mycards) >= midnum(hand))) || (cardStrength(mycards) >= 13 && (gettrashcards() < 22))){		//手札が5枚以上でA以上の強さのカードでカードが中央値より小さいか、捨てたカードが22枚未満でカードの強さが13以上なら温存
		}else{
			hand.removeAll(mycards);
	  		s.insertAll(mycards);
			trashcards += mycards.size();
		}
	  return true;
	}
	return true;

}

bool Group6::approve(const GameStatus & gstat){
  CardSet pl;
  pl = gstat.pile;
  if(!(pl == mymemory)){
    mymemory = pl;
    trashcards += pl.size();       //捨てられたカード枚数をプラス
  }
  return true;
}


/*
 * 順序関係 compareCards を使うナイーヴ naive なソート．
 * 枚数は少ないので，効率は気にしない．
 */
void Group6::sort(bool ascending) {
	for(int i = 0; i+1 < hand.size(); i++) {
		for(int j = i+1; j < hand.size(); j++) {
			if ( (ascending && cardGreaterThan(hand[i], hand[j]))
					|| (!ascending && cardLessThan(hand[i], hand[j])) ) {
				Card t = hand[i];
				hand[i] = hand[j];
				hand[j] = t;
			}
		}
	}
}

//   Returns true if and only if the left CardSet is either correct and stronger than
// the right one, or the right one is an illegal set.
//
bool Group6::cardsStrongerThan(const CardSet & left, const CardSet & right) {      //左<右ならtrue
	int leftRank, rightRank;

	// regarded as "pass"
	if (left.isEmpty() )
		return false;

	// left is an illegal set
	if (!checkRankUniqueness(left))
		return false;
	if ( left.size() >= 5 )
		return true;

	// left always wins
	if ( right.isEmpty() )
		return true;

	// right is an illegal set.
	if (!checkRankUniqueness(right))
		return true;
	if ( right.size() >= 5 )
		return true;

	// the number of cards of the left set must be match to that of the right one.
	if ( left.size() != right.size() )
		return false;


	leftRank = cardStrength(left);
	rightRank = cardStrength(right);

	if ( leftRank > rightRank )
		return true;
	else
		return false;
}

int Group6::cardStrength(const CardSet & cs) {
	int i;
	if ( cs.isEmpty() )
		return 0;

	if ( cs.size() == 1 && cs[0].isJoker() ) {
		return cardStrength(cs[0]);
	}
  	for (i = 0; i < cs.size(); i++) {
	  if (!cs[i].isJoker()) {
		  break;
	  }
	}
	return cardStrength(cs[i]);
}

bool Group6::checkRankUniqueness(const CardSet & cs) {
	int rank = 0;

	if (cs.size() == 0)
		return false;

	if ( cs.size() == 1 && cs[0].isJoker() )
		return true;

	for (int i = 0; i < cs.size(); i++) {
	  if (cs[i].isJoker() )
		  continue;  // Jkrをスキップ
	  if ( rank == 0 ) {
		  rank = cs[i].getNumber();
	  } else if ( rank != cs[i].getNumber() ) {
	    return false;
	  }
	}
	return true;
}

double Group6::myaverage(const CardSet & cset) {
	double sum = 0;
	if ( cset.size() == 0 )
		return sum;
	for(int i = 0; i < cset.size(); i++) {
		sum += cardStrength(cset[i]);
	}
	return sum/cset.size();
}

int Group6::cardStrength(const Card & c) {
	if ( c.isJoker() )
		return 18;
  	if ( c.getNumber() <= 2 )
  		return c.getNumber() + 13;
	return c.getNumber();
}

CardSet & Group6::findSmallestAcceptable(const GameStatus & gstate, const CardSet & cs, CardSet & mycs) {
	// assumes the hand is sorted in ascending order
	mycs.makeEmpty();
	int cssize = cs.size();						//場のカードの枚数
	int csstrength = cardStrength(cs);			//場のカードの強さ
	int i, n;
//	if ( cssize > 1 )							//同時出しならなんか出力する
//		std::cout << "multiple cards!!!" << std::endl;
	for(i = 0; i < hand.size(); ) {
		if ( cardStrength(hand[i]) <= csstrength ) {		//手札が場のカード以下の弱さだったらカウンタを更新する
			++i;
			continue;
		}
		//ここで場のカードよりは強い
		for(n = 1; i + n < hand.size(); ) {					//手札のサイズより小さい間
			if ( /*hand[i+n].isJoker()						
				||*/ (cardStrength(hand[i + n]) == cardStrength(hand[i])) ) {		//最弱のより+nした手札がジョーカーか同じ強さのやつであればnを増やしてcontinueそうでなければループを抜ける
				++n;															//要するに複数枚の時はその枚数を数えてnに入れる
				continue;
			}
			break;
		}
		if ( cssize <= n ) {			//場のカード数<=最弱のカード数
			
			if( cssize == n){
				for(int j = i; j < i + n; j++)
						mycs.insert(hand[j]);
			}else if ( (cssize != 0) && (cssize < n) 
					&& ((gstate.numCards[gstate.leaderIndex] <= 5) || (cardStrength(hand[i]) > 12)) ){	//既に場にカードがあるときかつリーダーのカード枚数が5枚以下nを場のカード数に合わせる
					n = cssize;
					
					for(int j = i; j < i + n; j++)
						mycs.insert(hand[j]);
					}else if( cssize == 0 ){
			for(int j = i; j < i + n; j++)
				mycs.insert(hand[j]);
			}
			if( (cssize == 0) || (n==cssize) )
			return mycs;
		}
		i += n;							//数字自体は強いけどカードの枚数が足りない時
	}
	return mycs;   // empty card set.
}

CardSet & Group6::StrongestCards(CardSet & mycs) {
  mycs.makeEmpty();
  int i=0, n = 0;
  
  i = hand.size() - 1;
  for(n = 1; i - n < 0; ) {					//手札全部の間
    if ( (cardStrength(hand[i - n]) == cardStrength(hand[i])) ) {		//最強のより+nした手札が同じ強さのやつであればnを減らしてcontinueそうでなければループを抜ける
      ++n;															//要するに複数枚の時はその枚数を数えてnに入れる
      continue;
    }
    break;
  }
  for(int j = i;j > i - n; j--){
    mycs.insert(hand[j]);
  }
  return mycs;
}
  

int Group6::mycardsize(CardSet mycs){             //手札の同じカードが何枚あるか(バグって無限ループに入るから使う機会がない)(そもそもCardset::sizeで代用できるから必要ない)
  int i, n;
  for(i = 0; i < hand.size();){
    if(cardStrength(hand[i]) == cardStrength(mycs)){
      ++i;
      continue;
    }
    for(n = 1; i + n < hand.size(); ) {					//手札のサイズより小さい間
      if ( hand[i+n].isJoker()						
	   || (cardStrength(hand[i + n]) == cardStrength(hand[i])) ) {		//最弱のより+nした手札がジョーカーか同じ強さのやつであればnを増やしてcontinueそうでなければループを抜ける
	++n;															//要するに複数枚の時はその枚数を数えてnに入れる
	continue;
      }
      break;
    }
  }
  return n;
}

double Group6::numcardAve(const GameStatus & gstate){		//それぞれのプレイヤーのカード枚数の平均より
	int sum=0;
	double ave;
	for(int i = 0;i<gstate.numPlayers;i++){
		sum += gstate.numCards[i];
	}
	ave = (double)sum/(double)gstate.numPlayers;
	return ave;
}

int Group6::smallestNumCard(const GameStatus & gstate){
	int min = 100;
	
	for(int i = 0;i<gstate.numPlayers; i++){
		if(min > gstate.numCards[i]){
			min = gstate.numCards[i];
		}
	}
	return min;
}

int Group6::midnum(const CardSet & cards){    //中央値
  int cssize=0, mid=0;
  cssize = cards.size();
  mid = cssize/2;
  return cardStrength(hand[mid]);
}

int Group6::gettrashcards(void){
	return trashcards;
}


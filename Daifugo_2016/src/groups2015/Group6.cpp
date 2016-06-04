/*
 *  Group6.cpp
 *  PlayingCard
 *
 *
 */
#include <groups2015/Group6.h>
#include <iostream>
#include <string>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"
using namespace grp2015;

void Group6::ready() {
	memory.makeEmpty(); //memory.clear();
	trump.makeEmpty(); //trump.clear();
}
bool Group6::follow(const GameStatus & gstat, CardSet & s) {
  CardSet pile(gstat.pile);
  Card tmp, pcard;
  int work[hand.size()], i, /* samem, */ psize, pnum, max, maxi, sameflg=0, num, c=0;
  //  std::cout << gstat << std::endl;
  s.makeEmpty(); //clear();
  sortInHand(hand);
  // std::cout << "( " << inHand() << " )" << std::endl;

  //相手の残り枚数を数える
  for(i=0;i<gstat.numPlayers;i++)
    c = gstat.numCards[i] + c;
  c = c - hand.size();
  if(c<2){ //相手の残り枚数が一枚のときの処理(二人だけの時)
    for(i=0;i<hand.size();i++)
      work[i]=0;
    if(pile.isEmpty()){
      if(sameRank(work,3)==1){
	for(i=0;i<hand.size();i++){
	  if(work[i]>0){
	    inHand().pickup(tmp,i);
	    s.insert(tmp);
	  }
	  if(s.size()==3)
	    return true;
	}    
    //   for(i=0;i<hand.size();i++)
      num = hand[0].getNumber();
      inHand().pickup(tmp, 0);
      s.insert(tmp);
      return true;
      }
 
      if(sameRank(work,2)==1)
	// for(i = 0;i<hand.size();i++)
	//	std::cout << "( " << work[i]<< " )" << std::endl;
	for(i=0;i<hand.size();i++){
	  if(work[i]>0){
	    inHand().pickup(tmp,i);
	    s.insert(tmp);
	  }
	  if(s.size()==2)
	    return true;
	}    
    //   for(i=0;i<hand.size();i++)
      num = hand[0].getNumber();
      inHand().pickup(tmp, 0);
      s.insert(tmp);
      return true;
    }
  }
    

  if(pile.isEmpty()){ //場札がないとき
    num = hand[hand.size()-1].getNumber();
    inHand().pickup(tmp, hand.size()-1);//小さいものから選ぶ
    s.insert(tmp);
    for(i = hand.size()-1; num == hand[i].getNumber();i--){
      inHand().pickup(tmp, i);//小さいものから選ぶ(複数枚)
      s.insert(tmp);
    }
    return true;
  } 
  
  psize = pile.size();//場札の枚数
  pnum = pile.pickup(pcard,0);//場札の値
  pnum = (pcard.getNumber() + 23) % 13;
  for(i=0;i<hand.size();i++){ //簡単な重み付け
    if(pcard.isJoker()){ //場の札がジョーカーの時は最小のものを出す
      inHand().pickup(tmp, 0); 
      s.insert(tmp);
      return true;
    }
    if(hand[i].isJoker())//ジョーカーの重み
      work[i]=13;
    else//カードの数字ごとの重み
      work[i] = (hand[i].getNumber() + 23) % 13;
    if(work[i] <= pnum){
      work[i] = 0;
    }else{
      if(c>=2)
	work[i] = 14 - work[i];
    }
  }
  
  if(hand.size()>4 && c>=2){//手札が4枚より多いときは手札の中で
    for(i=0;i<2;i++)
      work[i] = 0;
  }
  if(psize>1)
    sameflg = sameRank(work,psize);//複数枚の処理
  else
    noSame(work);//場札が一枚の時の処理
  //   for(i = 0;i<hand.size();i++)
  //   std::cout << "( " << work[i]<< " )" << std::endl;
  //	std::cout << "***********" << std::endl;

//重みが最大のものを選ぶ
  max = work[hand.size()-1];
  maxi = hand.size()-1;
  for(i=maxi-1;i>=0;i--)
    if(work[i] > max){
      max = work[i];
      maxi = i;
    }
  //   std::cout << maxi<< std::endl;
  if(sameflg == 1){//複数枚の処理
    for(i=maxi;s.size() < psize;i--){
      inHand().pickup(tmp,i);
      s.insert(tmp);
    }
  }else{
    inHand().pickup(tmp, maxi);
    s.insert(tmp);
  }
  return true;
  
}

/*
 * ソートに使う順序関数の例．
 * 自分のならべたい順序の定義を作成する．
 * これはカード同士の比較であり，カードの集合同士の比較ではない．
 */
bool Group6::cardLessThan(const Card & c1, const Card & c2) {
  int card1, card2;
  
  if(c1.isJoker() == true)
    return false;
  if(c2.isJoker() == true)
    return true;
  card1 = (c1.getNumber() + 23) % 13;
  card2 = (c2.getNumber() + 23) % 13;
  if(card1 < card2)
    return true;
  return false;
}

/*
 * 順序関数と整合性のある同値（等しい）関係．
 */
bool Group6::cardEquals(const Card & c1, const Card & c2) {
	return !cardLessThan(c1, c2) && !cardLessThan(c2, c1);
}

/*
 * 順序関係 compareCards を使うナイーヴな naive ソートの例．
 * 枚数は少ないので，効率は気にしない．
 */
void Group6::sortInHand(CardSet & cset) {
	for(int i = 0; i+1 < cset.size(); i++) {
		for(int j = i+1; j < cset.size(); j++) {
			if ( cardLessThan(cset[i], cset[j]) ) {
				Card t = cset[i];
				cset[i] = cset[j];
				cset[j] = t;
			}
		}
	}
}
/*
 * 複数枚の処理
 * 場の枚数と同じ枚数のものに点数を与える
 */
int Group6::sameRank(int work[], int psize){
  int i, j, same, flg;
  
  flg = 0;
  for(i = 0; i < hand.size()-1; i++){
    same = 1;
    for(j = i+1; hand[i].getNumber() == hand[j].getNumber(); j++)
      same++;
    if(psize == same){
      for(; i < j; i++)
	if(hand.size() <= psize+1)
	  work[i] = work[i] + 20;
	else if((hand[i].getNumber() + 23) % 13 < 10)
	    work[i] = work[i] + 20;
      flg = 1;
    }
  }
  
  return flg;
}
/*
 * 複数枚の処理
 * 場の枚数が一枚の時に複数枚のものを出さないようにする
 */
void Group6::noSame(int work[]){
  int i, j, same;
  
  for(i = 0; i < hand.size()-1; i++){
    same = 1;
    if(((hand[i].getNumber()+23)%13) <11){
      for(j = i+1; hand[i].getNumber() == hand[j].getNumber(); j++)
	same++;
      if(same>1){
	for(; i < j; i++)
	  work[i] = 0;
      }
    }
  }
}


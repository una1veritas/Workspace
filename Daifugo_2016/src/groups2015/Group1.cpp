/*
 *  Group1.cpp
 *  PlayingCard
 *
 *  Created by 下薗 真一 on 09/04/12.
 *  Modified by Kazutaka Shimada on 09/04/21.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include <groups2015/Group1.h>
#include <iostream>
#include <string>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"

using namespace grp2015;

void Group1::ready() {
  memory.makeEmpty(); //memory.clear();
  trump.makeEmpty(); //trump.clear();
}

bool Group1::approve(const GameStatus & gstat) {
  memory.insert(gstat.pile); //メモリに場札のカードを追加
  /*
     std::cout << "my ID =" << " " << getId() << std::endl;
     for(int i = 0; i < gstat.nofPlayers; i++){
     std::cout << "Player" << i << "'s CardNum ="<< " " << gstat.nofCards[i] << std::endl;
     }
  //    std::cout << "\nmemory" << std::endl;
  //    memory.print();
  */
  return true;
}


bool Group1::follow(const GameStatus & gstat, CardSet & s) {
  CardSet pile(gstat.pile);
  Card bafuda, tmp;
  int /* pos = 0,*/ i = 0, j= 0, k = 0;
  int pileCardNum = pile.size();
  bool syoban = true; //初盤フラグ
  bool jkrFlag = false; //ジョーカーフラグ
  int sameCardNum; //同じ数字のカードの枚数
  int cardNum; //場札に出してよいとするカードの上限
  int markedPlayerNum = 0; //手札が規定枚数以下のプレイヤ人数
  int numoftehuda =4; //初盤判定の基準

  std::cout << gstat << std::endl;
  s.makeEmpty(); //clear();
  sortInHand();
  std::cout << "( " << inHand() << " )" << std::endl;

  //ジョーカーを持ってるかどうか
  tmp = inHand()[0];
  if(tmp.getRank() == 0) jkrFlag = true;

  //初盤かどうか判定
  for(i=0;i<=gstat.numPlayers;i++) {
    if(gstat.numCards[i] < numoftehuda) {
      markedPlayerNum++;
      if((markedPlayerNum > 1 && inHand().size() >= numoftehuda) || (markedPlayerNum > 2 && inHand().size() < numoftehuda)) {
        syoban = false;
        break;
      }
    }
  }

  if(syoban) std::cout << "syoban\n";
  else std::cout << "not syoban\n";

  //強いカードを残す
  if(!syoban) { //初盤じゃない(手札４枚以下のプレイヤが居る)
    cardNum = inHand().size(); //全部が探索対象
  } else {  //初盤は一番強いものがペアであればそれを除外する
    sameCardNum = numofSameCard(inHand()[0]); //そのペア以外のものが探索対象
    if(jkrFlag) sameCardNum = 1;
    cardNum = inHand().size() - sameCardNum;
  }


  //複数枚か判定
  switch(pileCardNum) {
    case 0: //自分が親となった（=場札がない）時
      if(syoban) {
        i = inHand().size() - 1;
        sameCardNum = numofSameCard(inHand()[i]);
        if(sameCardNum > 1) i -= sameCardNum;
        inHand().pickup(tmp, i);
        s.insert(tmp);
        return true;
      } else {
        for(k=4;k>0;k--) { //4枚のペアから存在するかどうか探して、あれば出す
          for(i=inHand().size()-1;i>=0;i--) {
            sameCardNum = numofSameCard(inHand()[i]);
            if(sameCardNum == k) {
              for(j=0;j<sameCardNum;j++) {
                inHand().pickup(tmp, i-j);
                s.insert(tmp);
              }
              return true;
            }
          }
        }
        for(k=4;k>1;k--) {
          for(i=inHand().size()-1;i>0;i--) { //ジョーカーも含めて出す場合
            if(cardLessThan(pile[0], inHand()[i])) {
              sameCardNum  = numofSameCard(inHand()[i]);
              if(sameCardNum == k && jkrFlag) { //一枚足りなくて、ジョーカーがある時
                for(j=0;j<sameCardNum;j++) {
                  inHand().pickup(tmp, i-j);
                  s.insert(tmp);
                }
                inHand().pickup(tmp, 0); //一番最初のカード(=ジョーカー)を出す
                s.insert(tmp);
                jkrFlag = false;
                return true;
              }
            }
          }
        }
      }
    case 1:
      bafuda = pile[0]; //場札リストの最初（場札の強いカード”1枚”）をbafudaにいれる

      for(i=inHand().size()-1;i>=inHand().size()-cardNum;i--){     //手札の最弱から強い札まで
        tmp = inHand()[i];
        if(cardLessThan(bafuda, tmp)){     //場札のほうが弱いとき
          sameCardNum = numofSameCard(tmp); //ペア以上のカードかどうか
          if(sameCardNum  > 1 && syoban) { //ペア以上かつ初盤なら
            i -= sameCardNum - 1;
            continue;
          }
          inHand().pickup(tmp,i);    //場札より少し強いカードを手札から選ぶ
          s.insert(tmp);
          return true;
        }
      }	    
      break;
    case 2: //2枚ペアの時
std::cout << "case 2\n";
      for(i=inHand().size()-1;i>inHand().size()-cardNum-1;i--) { //手札の中でpileのカードセットより強いカードを選んで出す
        if(cardLessThan(pile[0], inHand()[i])) {
          sameCardNum  = numofSameCard(inHand()[i]);
          if(sameCardNum == 2 || (sameCardNum >= 2 && !syoban)) {
            for(j=0;j<2;j++) {
              inHand().pickup(tmp, i-j);
              s.insert(tmp);
            }
            return true;
          }
        }
      }
      break;
    case 3: //3枚ペアの時
      for(i=inHand().size()-1;i>inHand().size()-cardNum-1;i--) { //手札の中でpileのカードセットより強いカードを選んで出す
        if(cardLessThan(pile[0], inHand()[i])) {
          sameCardNum  = numofSameCard(inHand()[i]);
          if(sameCardNum == 3 || (sameCardNum >= 3 && !syoban)) {
            for(j=0;j<3;j++) {
              inHand().pickup(tmp, i-j);
              s.insert(tmp);
            }
            return true;
          }
        }
      }
      if(syoban) break;
      for(i=inHand().size()-1;i>0;i--) { //ジョーカーも含めて出す場合
        if(cardLessThan(pile[0], inHand()[i])) {
          sameCardNum  = numofSameCard(inHand()[i]);
          if(sameCardNum == 2 && jkrFlag) { //一枚足りなくて、ジョーカーがある時
            for(j=0;j<sameCardNum;j++) {
              inHand().pickup(tmp, i-j);
              s.insert(tmp);
            }
            inHand().pickup(tmp, 0); //一番最初のカード(=ジョーカー)を出す
            s.insert(tmp);
            jkrFlag = false;
            return true;
          }
        }
      }
      break;
    case 4: //4枚ペアの時
std::cout << "case4\n";
      for(i=inHand().size()-1;i>inHand().size()-cardNum-1;i--) { //手札の中でpileのカードセットより強いカードを選んで出す
        if(cardLessThan(pile[0], inHand()[i])) {
          sameCardNum  = numofSameCard(inHand()[i]);
          if(sameCardNum == 4) {
            for(j=0;j<4;j++) {
              inHand().pickup(tmp, i-j);
              s.insert(tmp);
            }
            return true;
          }
          if(sameCardNum == 3 && jkrFlag) { //一枚足りなくて、ジョーカーがある時
            for(j=0;j<sameCardNum;j++) {
              inHand().pickup(tmp, i-j);
              s.insert(tmp);
            }
            inHand().pickup(tmp, 0); //一番最初のカード(=ジョーカー)を出す
            s.insert(tmp);
            jkrFlag = false;
            return true;
          }
        }
      }
      break;
  }
  return false;
  // the card idential to tmp is already removed from the hand. 
  // cardSetOfSameRanks(s, pile.size());
  // たとえば、複数枚のカードを探す関数。ただしこの関数は未実装。
  // 現状ではこの follow は Player.cc のものと等価
}

/*
 * ソートに使う順序関数の例．
 * 自分のならべたい順序の定義を作成する．
 * これはカード同士の比較であり，カードの集合同士の比較ではない．
 */
bool Group1::cardLessThan(const Card & c1, const Card & c2) {
  int c1Num = c1.getNumber();
  int c2Num = c2.getNumber();
  if(c2Num == 0) return true;
  if(c1Num == 0) return false;

  if(c1Num == 2) return false;
  if(c2Num == 2) return true;

  if(c1Num == 1) {
    if(c2Num == 2) return true;
    return false;
  }
  if(c2Num == 1) {
    if(c1Num == 2) return false;
    return true;
  }

  if ( c1.getNumber() < c2.getNumber() )
    return true;
  return false;
}

/*
 * 順序関数と整合性のある同値（等しい）関係．
 */
bool Group1::cardEquals(const Card & c1, const Card & c2) {
  return !cardLessThan(c1, c2) && !cardLessThan(c2, c1);
}

/*
 * 順序関係 compareCards を使うナイーヴな naive ソートの例．
 * 枚数は少ないので，効率は気にしない．
 */
void Group1::sortInHand(void) {
  for(int i = 0; i+1 < hand.size(); i++) {
    for(int j = i+1; j < hand.size(); j++) {
      if ( cardLessThan(hand[i], hand[j]) ) {
        Card t = hand[i];
        hand[i] = hand[j];
        hand[j] = t;
      }
    }
  }
}

/*
 *cardで渡されたものと同じ数字のカードの枚数を返す
 */
int Group1::numofSameCard(const Card & card) {
  int num = 0;
  int i = 0;

  for(i=0;i<inHand().size();i++) {
    if(!cardEquals(inHand()[i], card)) continue;
    num++;
  }
  return num;
}

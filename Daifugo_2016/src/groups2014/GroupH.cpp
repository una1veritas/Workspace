#include <iostream>
#include <string>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"
#include "GroupH.h"

using namespace grp2014;

#define MAXCARD 17

GroupH::GroupH(const char * s) : Player(s) {}

bool GroupH::approve(const CardSet & pile, int numCards[]) {
  memory.insertAll(pile);
  // memoryには他のプレイヤーが場に出したカードがストックされる
  
  // 自分自身のidは getId() で獲得できる
  // std::cout << "I am "<< getName() << " " << getId() << std::endl;

#ifdef DEBUG
  for (int i=0; numCards[i] != 99; i++){
    // numCardsには自分を含めたそれぞれの持ち札の数が格納されている
    // 配列の最後には終端検知のため"99"という値が入っている
    // numCardsの添字はプレイヤーのidを表す
     std::cout << i << " " << numCards[i] << std::endl;
  }
#endif
  return true;
  
}

bool GroupH::follow(const CardSet & pile, CardSet & s) {
  Card tmp;
  CardSet tmp1;
  int i, j, k, l, mhsize, cmpp1 = 0, cmpp2 = 0, cmpp3 = 0, pcard, pcnum, large;
  double per;
  int cnum[MAXCARD], tablec[MAXCARD];
  int flag = 1; //, lowf = 1;

  /* 初期化 */
  s.clear();
  tmp1.clear();
  i = 0;
  j = 0;
  k = 0;
  l = 0;

  pcard = pile[0].getRank();
  pcnum = pile.size();
  mhsize = inHand().size();

  /* カードをソート */
  sort();
  /* カードのそれぞれの数字における枚数を管理する配列の初期化 */
  for(i = 0; i < MAXCARD; i++){
    cnum[i] = 0;
    tablec[i] = 0;
  }

  /* 場に出ているカードの数字における枚数の管理 */
  for(i = 0; i < memory.size(); i++){
      k = memory[i].getRank();
      if(k == 0){
        tablec[16]++;
      }
      else if(k == 1){
        tablec[14]++;
      }
      else if(k == 2){
        tablec[15]++;
      }
      else{
        tablec[k]++;
      }
    }

  /* カードのそれぞれの数字における枚数の管理 */
  for(i = 0; i < inHand().size(); i++){
    k = inHand()[i].getRank();
    if(k == 0){
      cnum[16]++;
    }
    else if(k == 1){
      cnum[14]++;
    }
    else if(k == 2){
      cnum[15]++;
    }
    else{
      cnum[k]++;
    }
  }

  /* カードを3つの階級に分け優先などを決める */
  for(i = 3; i <= 6; i++){
    cmpp1 += cnum[i];
/*
    if(cnum[i] >= 3){
    	lowf = 0;
    }
 */
  }
  for(i = 7; i <= 10; i++){
    cmpp2 += cnum[i];
  }
  for(i = 11; i <= 15; i++){
    cmpp3 += cnum[i];
  }
  /* 手札の中で最大のものを保持 */
  large = 16;
  for(i = 16; i > 2; i--){
    if(cnum[i] >= 1){
      large = i;
      break;
    }
  }

  j = 0;
  for(i = large + 1; i <= 16; i++){
	  if(i == 16){
		  if(tablec[i] == 1){
			  j++;
		  }
	  }
	  else{
		  if(tablec[i] == 4){
			  j++;
		  }
	  }
  }
  if((16 - large) == j){
	  flag = 0;
  }


  /* 自分が始めにカードを出す場合 */
  if (pile.isEmpty()) {
    if(mhsize >= 4){
      for(i = 3; i < 10; i++){
	if(cnum[i] >= 2){
	  for(j = 3; j < i; j++){
	    l = l + cnum[j];
	  }
	  for(j = 0; j < cnum[i]; j++){
	    inHand().pickup(tmp, l);
	    s.insert(tmp);
#ifdef DEBUG
	    s.print();
#endif
	  }
	  return true;
	}
      }
    }
    else{
      for(i = 3; i < 15; i++){
	if(cnum[i] >= 2){
	  for(j = 3; j < i; j++){
	    l = l + cnum[j];
	  }
	  for(j = 0; j < cnum[i]; j++){
	    inHand().pickup(tmp, l);
	    s.insert(tmp);
#ifdef DEBUG
	    s.print();
#endif
	  }
	  return true;
	}
      }
    }
    inHand().pickup(tmp, 0);
    s.insert(tmp);
#ifdef DEBUG
    s.print();
#endif
    return true;
  }
  /* 場に出ているカードから自分が出すカード決定する場合 */
  else {
    if(pcard == 0){
      pcard = 16;
    }
    else if(pcard == 1){
      pcard = 14;
    }
    else if(pcard == 2){
      pcard =15;
    }

    if(mhsize > 5 || cmpp1 != 0){
        per = (double)cmpp3 / (double)cmpp1;
    }
    else{
      per = 1;
    }

    for(i = pcard + 1; i <= 16; i++){
      if(cnum[i] >= pcnum && !(i <= 8 && cnum[i] > pcnum)){
	if((i < 12 || per > 0.9) && !(i == large && cnum[i] == pcnum && mhsize > 3 && flag == 0)){
	  for(j = 3; j < i; j++){
	    l = l + cnum[j];
	  }
	  for(j = 0; j < pcnum; j++){
	    inHand().pickup(tmp, l);
	    s.insert(tmp);
	  }
	}
	return true;
      }
      else if(large == 16 && cnum[15] > 0 && i != 16){
	for(i = pcard; i <= 15; i++){
	  if(cnum[i] == pcnum - 1){
	    for(j = 3; j < i; j++){
	      l = l + cnum[j];
	    }
	    for(j = 0; j < pcnum - 1; j++){
	      inHand().pickup(tmp, l);
	      s.insert(tmp);
	    }
	    l = inHand().size();
	    inHand().pickup(tmp, l - 1);
	    return true;
	  }
	}
      }
    }
  }

  return true;
}


// 手札を bubble sort
void GroupH::sort(){
  int i,j;
  Card tmp;
  for(i = 0; i < inHand().size() ; i++){
    for(j = inHand().size() ; j > i; j--){
      if( cardGreaterThan( inHand()[j-1] ,inHand()[j])){
	inHand().pickup(tmp, j-1);
	inHand().insert(tmp);
      }
    }
  }
  
#ifdef DEBUG
inHand().print();
#endif
}


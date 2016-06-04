#include "GroupE.h"

#include <iostream>
#include <string>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"

using namespace grp2013;

GroupE::GroupE(const char * s) : Player(s) {}

bool GroupE::approve(const CardSet & pile, const int numCards[]){
  memory.insert(pile);



  return false;
}

bool GroupE::follow(const CardSet & pile, CardSet & s){
  	Card tmp[53];
	Card pic;
	Card sub;
	CardSet pass;
	CardSet ret[13];
	int limit_i = size();
	int limit_j = pile.size();
	int count = -1;

	s.makeEmpty();
	SearchCards();//揃っているカードを分類する


	switch (limit_j){
	case 0://リーダを出す
//2ペアで一番弱いカードを出す(3,4,5,6)のシングルカードの方が優先度高い)
	  if (((two_flag > 0) && (singlcard[3] != 1 && singlcard[4] != 1 && singlcard[5] != 1)) || (twocards[3] == 1 || twocards[4] == 1 || twocards[5] == 1)  ){//1と2の2ペアはリーダとしては出さない
	    for (int i = 3; i < 14; i++){//2ペアで出せるカードの候補を求める
	      if (twocards[i] == 1){
		ret[i] = PickUpCard(i);//2ペアのカードセットをつくる
		tmp[++count] = ret[i].at(0);
	      }
	    }
	    if (count != -1){//2ペアで出す
		pic = min_card(tmp,count+1);
		inHand().remove(ret[pic.getRank()].at(0));
		inHand().remove(ret[pic.getRank()].at(1));
		s = ret[pic.getRank()];
		two_flag--;
		return true;
	    }
	  }
//3カードで一番弱いカードを出す
	  if (((three_flag > 0) && (singlcard[3] != 1 && singlcard[4] != 1 && singlcard[5] != 1)) || (threecards[3] == 1 || threecards[4] == 1 || threecards[5] ==1)){
	    for (int i = 0; i < 14; i++){//3ペアで出せるカードの候補を求める
	      if (threecards[i] == 1 ){
		ret[i] = PickUpCard(i);//3ペアのカードセットをつくる
		tmp[++count] = ret[i].at(0);
	      }
	    }
	    if (count != -1){
		pic = min_card(tmp,count+1);
		inHand().remove(ret[pic.getRank()].at(0));
		inHand().remove(ret[pic.getRank()].at(1));
		inHand().remove(ret[pic.getRank()].at(2));
		s = ret[pic.getRank()];
		three_flag--;
		return true;
	    }
	    }
//4カードで一番弱いカードを出す
	  if (four_flag > 0){
	    for (int i = 0; i < 14; i++){//4ペアで出せるカードの候補を求める
	      if (((fourcards[i] == 1) && (singlcard[3] != 1 && singlcard[4] != 1 && singlcard[5] != 1)) || (fourcards[3] == 1 || fourcards[4] == 1 || fourcards[5] == 1)){
		ret[i] = PickUpCard(i);//4ペアのカードセットをつくる
		tmp[++count] = ret[i].at(0);
	      }
	    }
	    if (count != -1){
		pic = min_card(tmp,count+1);
		inHand().remove(ret[pic.getRank()].at(0));
		inHand().remove(ret[pic.getRank()].at(1));
		inHand().remove(ret[pic.getRank()].at(2));
		inHand().remove(ret[pic.getRank()].at(3));
		s = ret[pic.getRank()];
		four_flag--;
		return true;
	    }
	  }
//手持ちで一番弱いカードを1枚出す
	     for (int i = 0; i < limit_i; i++){
	      tmp[++count] = inHand().at(i);
	    }
	    pic = min_card(tmp,count+1);
	    inHand().remove(pic);
	    s.insert(pic);
	    return true;
	  break;
	case 1:	//一枚のとき
	    for (int i = 0; i < limit_i;i++){
	      if (pile.at(0).isJoker()){//jokerならパス
		s.insert(pass);
		return true;
	      }
	      if (cardGreaterThan(inHand().at(i), pile.at(0))
	    		  && (twocards[inHand().at(i).getRank()] != 1) ){//候補を選ぶ(ペアからは選ばない)
		tmp[++count] = inHand().at(i);
	      }
	    }
	    if (count != -1){
	      pic = min_card(tmp,count+1);
	      inHand().remove(pic);
	      s.insert(pic);
	    }
	    if (count == -1){
	      count = 0;
	      for (int i = 0; i < limit_i; i++){
		if ( cardGreaterThan(inHand().at(i), pile.at(0)) ){//候補を選ぶ(ペアからも選ぶ)
		  tmp[++count] = inHand().at(i);
		}
	      }	
	      if (count != -1){
		pic = min_card(tmp,count+1);
		inHand().remove(pic);
		s.insert(pic);
	      }
	    }
	
	    else s.insert(pass);
	    break;
	case 2://2枚だし
	  if(two_flag > 0){
	    for (int i = 0; i < 14; i++){//2ペアで出せるカードの候補を求める
	      sub.set(0,i);
	      if ( (twocards[i] == 1) && cardGreaterThan(sub, pile.at(0)) ){
		ret[i] = PickUpCard(i);//2ペアのカードセットをつくる
		tmp[++count] = ret[i].at(0);
	      }
	    }
	    if (count != -1){//2ペアで出す
		pic = min_card(tmp,count+1);
		inHand().remove(ret[pic.getRank()].at(0));
		inHand().remove(ret[pic.getRank()].at(1));
		s = ret[pic.getRank()];
		two_flag--;
		return true;
	    }
	  }
	    //出せる2ペアが無いとき
	  if (three_flag > 0){//3カードを分割して出す
	    for (int i = 0; i < 14; i++){//3カードで出せるカードの候補を求める
	      sub.set(0,i);
	      if ( (threecards[i] == 1) && cardGreaterThan(sub, pile.at(0)) ){
		ret[i] = PickUpCard(i);//3ペアのカードセットをつくる
		tmp[++count] = ret[i].at(0);
	      }
	    }
	    if (count != -1){
	        pic = min_card(tmp,count+1);//最小の候補を選ぶ
		inHand().remove(ret[pic.getRank()].at(0));//2枚だけ削除
		inHand().remove(ret[pic.getRank()].at(1));
		s.insert(ret[pic.getRank()].at(0));
		s.insert(ret[pic.getRank()].at(1));
		three_flag--;//3カードは１つ減る
		singlcard[pic.getRank()] = 1;//1枚のカードが残る
		return true;
	    }	    
	  }
	  //4カードを分割して出す
	  if (four_flag > 0){
	    for (int i = 0; i < 14; i++){//4カードで出せるカードの候補を求める
	      sub.set(0,i);
	      if ( (fourcards[i] == 1) && cardGreaterThan(sub, pile.at(0) ) ){
		ret[i] = PickUpCard(i);//4ペアのカードセットをつくる
		tmp[++count] = ret[i].at(0);
	      }
	    }
	    if (count != -1){
	        pic = min_card(tmp,count+1);//最小の候補を選ぶ
		inHand().remove(ret[pic.getRank()].at(0));//2枚だけ削除
		inHand().remove(ret[pic.getRank()].at(1));
		s.insert(ret[pic.getRank()].at(0));//2枚だけ出す
		s.insert(ret[pic.getRank()].at(1));
		four_flag--;//4カードは１つ減る
		two_flag++;
		twocards[pic.getRank()] = 1;//2ペアが1組増える
		return true;
	    }
	  }
	  if (joker_flag == 1){//ジョーカーで2ペアを作る
	    for (int i = 0; i < limit_i;i++){
	      if ( cardGreaterThan(inHand().at(i), pile.at(0))){
		tmp[++count] = inHand().at(i);
	      }
	    }
	    if (count != -1){
	      pic = min_card(tmp,count+1);
	      inHand().remove(pic);
	      inHand().remove(0);
	      s.insert(pic);
	      sub.set(4,0);//ジョーカー
	      s.insert(sub);
	      return true;
	    }
	  }
	  else 
	    s.insert(pass);
	  break;
//3カードで出す
	case 3:
	  if (three_flag > 0){
	    for (int i = 0; i < 14; i++){//3ペアで出せるカードの候補を求める
	      sub.set(0,i);
	      if ( (threecards[i] == 1) && cardGreaterThan(sub,pile.at(0)) ){
		ret[i] = PickUpCard(i);//3ペアのカードセットをつくる
		tmp[++count] = ret[i].at(0);
	      }
	    }
	    if (count != -1){
		pic = min_card(tmp,count+1);
		inHand().remove(ret[pic.getRank()].at(0));
		inHand().remove(ret[pic.getRank()].at(1));
		inHand().remove(ret[pic.getRank()].at(2));
		s = ret[pic.getRank()];
		three_flag--;
		return true;
	    }	
	    }
	  //4カードを分割して出す
	  if (four_flag > 0){
	    for (int i = 0; i < 14; i++){//4カードで出せるカードの候補を求める
	      sub.set(0,i);
	      if ( (fourcards[i] == 1) && cardGreaterThan(sub, pile.at(0)) ){
		ret[i] = PickUpCard(i);//4ペアのカードセットをつくる
		tmp[++count] = ret[i].at(0);
	      }
	    }
	    if (count != -1){
	        pic = min_card(tmp,count+1);//最小の候補を選ぶ
		inHand().remove(ret[pic.getRank()].at(0));//3枚だけ削除
		inHand().remove(ret[pic.getRank()].at(1));
		inHand().remove(ret[pic.getRank()].at(2));
		s.insert(ret[pic.getRank()].at(0));//3枚だけ出す
		s.insert(ret[pic.getRank()].at(1));
		s.insert(ret[pic.getRank()].at(2));
		four_flag--;//4カードは１つ減る
		singlcard[pic.getRank()] = 1;//1枚だけ残る
		return true;
	    }
	  }
	  if ((joker_flag == 1) && (two_flag > 0)){//ジョーカーを使って３カードを作る
	    for (int i = 0; i < 14; i++){//2ペアで出せるカードの候補を求める
	      sub.set(0,i);
	      if ( (twocards[i] == 1) && cardGreaterThan(sub, pile.at(0)) ){
		ret[i] = PickUpCard(i);//2ペアのカードセットをつくる
		tmp[++count] = ret[i].at(0);
	      }
	    }
	    if (count != -1){
	      pic = min_card(tmp,count+1);
	      inHand().remove(ret[pic.getRank()].at(0));
	      inHand().remove(ret[pic.getRank()].at(1));
	      inHand().remove(0);
	      sub.set(4,0);//ジョーカー
	      ret[pic.getRank()].insert(sub);//ジョーカーを加える
	      s = ret[pic.getRank()];
	      two_flag--;
	      return true;
	    }
	    s.insert(pass);
	    
	  }

	  break;
//4カードを返す
	case 4:
	  if (four_flag > 0){
	    for (int i = 0; i < 14; i++){//4ペアで出せるカードの候補を求める
	      sub.set(0,i);
	      if ( (fourcards[i] == 1) && cardGreaterThan(sub, pile.at(0)) ){
		ret[i] = PickUpCard(i);//4ペアのカードセットをつくる
		tmp[++count] = ret[i].at(0);
	      }
	    }
	    if (count != -1){
		pic = min_card(tmp,count+1);
		inHand().remove(ret[pic.getRank()].at(0));
		inHand().remove(ret[pic.getRank()].at(1));
		inHand().remove(ret[pic.getRank()].at(2));
		inHand().remove(ret[pic.getRank()].at(3));
		s = ret[pic.getRank()];
		four_flag--;
		return true;
	    }
	    else if (count == -1){
	      s.insert(pass);
	    }		
	    }
	  if ((joker_flag == 1) && (three_flag > 0)){//ジョーカーを使って4カードを作る
	    for (int i = 0; i < 14; i++){//3ペアで出せるカードの候補を求める
	      sub.set(0,i);
	      if ( (threecards[i] == 1) && cardGreaterThan(sub, pile.at(0)) ){
		ret[i] = PickUpCard(i);//3ペアのカードセットをつくる
		tmp[++count] = ret[i].at(0);
	      }
	    }
	    if (count != -1){
	      pic = min_card(tmp,count+1);
	      inHand().remove(ret[pic.getRank()].at(0));
	      inHand().remove(ret[pic.getRank()].at(1));
	      inHand().remove(ret[pic.getRank()].at(2));
	      inHand().remove(0);
	      sub.set(4,0);//ジョーカー
	      ret[pic.getRank()].insert(sub);//ジョーカーを加える
	      s = ret[pic.getRank()];
	      three_flag--;
	      return true;
	    }
	  }

	  break;
	default:
	  break;
	  }

	return true;
}

Card GroupE::min_card(Card c[],int limit){//一番弱いカードを返す
  Card min;
  min.set(4,0);

  for (int i=0;i<limit;i++){
    if( cardGreaterThan( min, c[i]) )
      min.set(c[i].getSuit(),c[i].getRank());
  }
 
  return min;

}

void GroupE::SearchCards(void){//揃っているカードを分ける
  int num[14] = {}; //0~13
  int limit = size();

  two_flag = 0;
  three_flag = 0;
  four_flag = 0;
  joker_flag = 0;


  for (int i = 0; i < 14; i++){
    singlcard[i] = 0;
    twocards[i] = 0;
    threecards[i] = 0;
    fourcards[i] = 0;
    for (int j = 0; j < limit; j++){
      if (inHand().at(j).getRank() == i) {
	num[i] += 1; //同じ数字のカードの枚数を数える
      }
    }
  }

  for (int i = 0; i < 14; i++){
    
    switch (num[i]){
    case 1:
      singlcard[i] = 1;
      break;
    case 2:
      twocards[i] = 1;
      two_flag++;
      break;
    case 3:
      threecards[i] = 1;
      three_flag++;
      break;
    case 4:
      fourcards[i] = 1;
      four_flag++;
      break;
    default:
      break;
    }
    if (num[0] == 1)
      joker_flag = 1;
  }
}

CardSet GroupE::PickUpCard(int a){//数字で指定されたカードをセットにして返す
  int limit = size();
  CardSet ret;
  for (int i = 0; i < limit; i++){
    if (inHand().at(i).getRank() == a){
      ret.insert(inHand().at(i));
    }
    }
  return ret;
}




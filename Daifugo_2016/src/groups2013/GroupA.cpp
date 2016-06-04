/*
 *  LittleThinkPlayer.cpp
 *  PlayingCard
 *
 *  Created by 松元 拓也 on 13/5/2~.
 *  Modified by Kazutaka Shimada on 09/04/21.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include <iostream>
#include <string>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"

#include "GroupA.h"
using namespace grp2013;

GroupA::GroupA(const char * s) : Player(s) { }

bool GroupA::approve(const CardSet & pile, int numCards[]) {

	//１ゲーム中に「場に出たカード」の管理
	int allHaveNum=0; 			//全てのプレーヤが持つカード枚数
	memory.insert(pile);			//「思考対象のカード」を「場に出たカード」に追加
	for (int i=0; numCards[i] != 99; i++) allHaveNum += numCards[i];	//全てのプレーヤが持つカード枚数の計算
	if((memory.size() + allHaveNum) > 53){	//全てのカード枚数が53枚を超えていたら異常(NEXT GAME?)
		memory.makeEmpty();		//「場に出たカード」リセット
		memory.insert(pile);		//「思考対象のカード」を「場に出たカード」に追加
	}
	
	//strongestCard=strongestNow(2);
/*
	std::cout << "your information is "<< playerName() << " " << getID() << std::endl;
	std::cout << "memory" << std::endl;
	sort(memory);	//「場に出たカード」整列
	memory.print();	//「場に出たカード」出力
	//std::cout << "Strongest Card is "; strongestCard.print(); std::cout << std::endl;
	for (int i=0; numCards[i] != 99; i++){
		std::cout << i << " " << numCards[i] << std::endl;
	}
*/
	return true;
	
}

bool GroupA::follow(const CardSet & pile, CardSet & s) {
	s.makeEmpty();
	/*
	std::cout << "\n---follow-Debug-s--" << std::endl;//
	std::cout << "mine" << std::endl;
*/	sort(inHand());		//「手持ちカード」整列
/*	inHand().print();	//「手持ちカード」出力
	std::cout <<"---follow-Debug-e--" << std::endl;//
*/

	thinkAndSetCard(pile,inHand(),s);	//「思考対象のカード」、「手持ちカード」から考え出すカードを「自分が出すカード」にセットする
	memory.insert(s); 	//「自分が出すカード」を「場に出たカード」に追加(ただし、自分の出すカードが絶対に正しいこと　TAは２枚出ししてくる)
	
	return true;
}

//並び替え
bool GroupA::sort(CardSet & mine) {
	Card strong;
	int size = mine.size();
	for(int i=0; i < size; i++){
		strong = mine[0];
		for(int j=1; j<size-i; j++){
			if(!cardGreaterThan(mine[j], strong) ){
				strong = mine[j];
			}
		}
		mine.remove(strong);
		mine.insert(strong);
	}
	return true;
}

void GroupA::calcRemains(CardSet dels){
  Card tmp;
CardSet remain;		//「場に出てないカード」
  remain.setupDeck();	//「場に出てないカード」にすべてのカードをセット
  for(int i=0; i < memory.size(); i++){	//「場に出てないカード」から「場に出たカード」を抜く
    remain.remove(memory[i]);
  }
	dels.print();
  for(int j=0; j < dels.size(); j++){
	remain.remove(dels[j]);
  }
  sort(remain);	//「場に出てないカード」整列
  //remain.print();
  remains.reset();
  remains.set(remain);
  remains.print();
}


void GroupA::calcRemains(){
  CardSet remain;		//「場に出てないカード」
  remain.setupDeck();	//「場に出てないカード」にすべてのカードをセット
  for(int i=0; i< memory.size(); i++){	//「場に出てないカード」から「場に出たカード」を抜く
    remain.remove(memory[i]);
  }
  sort(remain);		//「場に出てないカード」整列
  remains.reset();
  remains.set(remain);
  //printf("残り");
  //remains.print();
}

CardSet GroupA::strongestNow(int num){
	
  CardSet tmp;
	for(int i=remains.getSetSize()-1;i>=0;i--){
		if(remains.at(i).getLevel() >= num){
			return remains.at(i).getCards();
		}
	}
	return tmp;
}

bool GroupA::thinkAndSetCard(const CardSet & pile,CardSet & mine,CardSet & s){
	Card tmpCard;
	MyCard tmpMyCard;
	CardSet tmpCardSet,strongestCardSet;
	int count=0;
	bool flag=false;
	
	//自分のカードの分別
	sort(mine);
	mycards.reset();
	mycards.set(mine);
	printf("手持ち");
	//mycards.print();

	calcRemains();
	strongestCardSet=strongestNow(pile.size());

	//現在のゲームの最強カードの算出
	//printf("%d枚の最強カード　",pile.size());
	//strongestCardSet.print();

	  //
	count=0;

	for(int j=0;j<4;j++){
		strongestCardSet=strongestNow((pile.size()+j)%4);
		tmpCard=strongestCardSet[0];
		for(int i=0;i<mycards.getSetSize();i++){
			if(mycards.at(i).getrank()==tmpCard.getRank()){
				mycards.atSetWait(i,true);
	//			mycards.at(i).getCards().print();
				flag=true;
				
			}
		}
	}	
//	
//	calcRemains(mycards.at(i).getCards());
	
	
	printf("%d",count);
	mycards.atSetWait(0,true);
	mycards.atSetWait(mycards.getSetSize()-1,true);
	  
	  
	  //現在のゲームで出されたカード枚数が15枚(残り38枚)
	  if(memory.size()<=15){
	    for(int i=0;i<mycards.getSetSize();i++){
	      if((mycards.at(i).getrank()==13)||(mycards.at(i).getrank()<=2)){
		mycards.atSetWait(i,true);
	      }
	    }
	  }
	if(mycards.getWaitMyCardNum()==mycards.getSetSize()){
		for(int i=0;i<mycards.getSetSize();i++){
			mycards.atSetWait(i,false);
	    	}
	}
//	mycards.print();
	count=0;

	switch(pile.size()){
		case 0:
                        for(int i=mycards.getSetSize()-1;i>=0;i--){
				if((!(mycards.at(i).isWait()))){
					tmpCardSet=mycards.at(i).getCards();
				}
                        }
			for(int j=0;j<tmpCardSet.size();j++){
				tmpCard=tmpCardSet[j];
				s.insert(tmpCard);
				mine.remove(tmpCard);
			}
		break;

		case 1:
			for(int i=0;i<mycards.getSetSize();i++){
				if((!(mycards.at(i).isWait()))&&(mycards.at(i).getLevel()>=1)){
					tmpCardSet=mycards.at(i).getCards();
					for(int j=0;j<tmpCardSet.size();j++){
						tmpCard=tmpCardSet[j];
						if ( cardGreaterThan(tmpCard,pile[0]) ){
							s.insert(tmpCard);
							mine.remove(tmpCard);
							return true;
						}
					}
				}
			}
				
		break;
		
		case 2:
			for(int i=0;i<mycards.getSetSize();i++){
				if((!(mycards.at(i).isWait()))&&(mycards.at(i).getLevel()>=2)){
					tmpCardSet=mycards.at(i).getCards();
					for(int j=0;j<tmpCardSet.size();j++){
						tmpCard=tmpCardSet[j];
						if( cardGreaterThan(tmpCard, pile[0]) ){
							s.insert(tmpCard);
							mine.remove(tmpCard);
							count++;
							if(count==2)return true;
						}
					}
				}
			}

		break;
		case 3:
			for(int i=0;i<mycards.getSetSize();i++){
				if((!(mycards.at(i).isWait()))&&(mycards.at(i).getLevel()>=3)){
					tmpCardSet=mycards.at(i).getCards();
					for(int j=0;j<tmpCardSet.size();j++){
						tmpCard=tmpCardSet[j];
						if( cardGreaterThan(tmpCard, pile[0]) ){
                                                        s.insert(tmpCard);
							mine.remove(tmpCard);
							count++;
							if(count==3)return true;
						}
					}
				}
			}
		break;	
		
		default:
		break;
	}
	return true;
}

bool MyCard::set(Card c){
	bool okflag=true;
	for(int i=0;i<cards.size();i++){
		if(cards[i].getRank() != c.getRank()){
			okflag=false;
		}
	}
	if(okflag){
		cards.insert(c);
		return true;
	}else{
		return false;
	}
}

void MyCard::setWait(bool tf){
//printf("MyCard::setWait()");
	waiting=tf;
//printf("%x\n",&waiting);
}

bool MyCard::isWait(){
//printf("MyCard::isWait()");
//printf("%x\n",&waiting);
	return waiting;
}

void MyCard::reset(){
	cards.makeEmpty();
	waiting=false;
}
bool MyCard::equal(MyCard tgt){
  if(getLevel()!=tgt.getLevel()) return false;
  Card tmp1,tmp2;
  int count=0;
  for(int i=0;i<getLevel();i++){
    count=0;
    tmp1=getCards()[i];
    for(int j=0;j<tgt.getLevel();j++){
      tmp2=tgt.getCards()[j];
      if( GroupA::cardEquals(tmp1, tmp2) ) count++;
    }
  }
  if(count==(tgt.getLevel()*tgt.getLevel()))return true;
  return false;
}

int MyCard::getrank(){
	bool flag=true;
	for(int i=0;i<cards.size()-1;i++){
		if (cards[i].getRank() != cards[i+1].getRank() ) {
			flag=false;
		}
	}
	if(flag){
		return cards[0].getRank();
	}else{
		return -1;//
	}
}

bool MyCard::isJoker(){
	if ( cards.size() == 1 && cards[0].isJoker() ) return true;
	else return false;
}

MyCard MyCardSet::at(int i){
  return cards[i];
}

void MyCardSet::atSetWait(int i,bool tf){
  if(numcard>=i) return cards[i].setWait(tf);
}


bool MyCardSet::set(CardSet cardset){
	Card tmp;
	MyCard tmp2;
	for(int i=0; i<cardset.size(); i++){
		if( GroupA::cardGreaterThan(cardset[i], tmp) && (i!=0) ){
		  insert(tmp2);
		  tmp2.reset();
		}
		tmp = cardset[i];
		tmp2.set(tmp);
	}
	return insert(tmp2);
}

void MyCardSet::reset(){		//初期化
	for(int i=0;i<maxnumcard;i++) cards[i].reset();
	numcard = 0 ;
}

int MyCardSet::locate(MyCard target)
{
	for(int i = 0; i < numcard; i++)
	  if(target.equal(cards[i])){
	    return i;
	  }
	return -1;	// 見つからなかった
}

bool MyCardSet::insert(MyCard card){
	if(locate(card) >= 0)
		return false;
	
	cards[numcard] = card;
	numcard++;

	return true;
}

bool MyCardSet::remove(MyCard target){
	int pos;
	if((pos = locate(target)) < 0){
	  return false;
	}
	for(int i = pos + 1; i < numcard; i++){
	  cards[i-1] = cards[i];
	}
	numcard--;
	return true;
}


void MyCardSet::print(){	printf("カード---\n");
	for(int i=0;i<numcard;i++){
		if( cards[i].isWait()) printf("Wait "); else printf("not wait ");
		cards[i].getCards().print();
	}
	printf("------\n");
}

int MyCardSet::getWaitMyCardNum(){
  int count=0;
  for(int i=0;i<numcard;i++){
    if(cards[i].isWait()) count++;
  }
  return count;
}

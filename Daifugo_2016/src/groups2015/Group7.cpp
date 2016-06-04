#include <groups2015/Group7.h>
#include <iostream>
#include <string>
#include <stdio.h>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"

using namespace grp2015;

#define TAIKAI_HONBAN

void Group7::ready() {
	memory.makeEmpty(); //memory.clear();
	trump.makeEmpty(); //trump.clear();
}

//使用されたカードをmemory記憶する
bool Group7::approve(const GameStatus & gstat){
  CardSet pile(gstat.pile);
  memory.insert(pile);
  memorysort();

  return true;
}

bool Group7::follow(const GameStatus & gstat, CardSet & s) {
	CardSet pile(gstat.pile);
	Card tmp, tmp2;
	int /* k, */ i, num, c, c2, flag;

	s.makeEmpty();
	sort();//手札を弱いのが左、強いのが右になるようにソート
#ifndef TAIKAI_HONBAN
	std::cout << gstat << std::endl;
	std::cout << "( " << inHand() << " )" << std::endl;
#endif
	
	if(pile.size() != 0){//親でない時
		tmp2 = pile[0];
		//手札を弱い方から見ていく
		for(i = 0; i < hand.size(); i++){
		//場のカードよりも強く、場に出ている枚数よりも多く持っているカードを探す
	    		if(cardGreaterThan(hand[i], tmp2) && cardNum(hand[i].getNumber(), hand) >= pile.size()){
	       			//最初の条件を満たした上で場のカードが２枚出しじゃない時に
				//手札の選んだ数の枚数が２枚ではなかったら決定
				if(cardNum(hand[i].getNumber(), hand) != 2 && pile.size() != 2)
		 			break;
	       		
				//最初の条件を満たした上で場のカードが２枚だしの時に
				//手札の選んだ数が２枚なら決定
				if(cardNum(hand[i].getNumber(), hand) == 2 && pile.size() == 2)
		 			break;

				//２は１枚だしの時に使う
	       			if(hand[i].getNumber() == 2 && pile.size() == 1)
		 			break;
			
				//手札が５枚以下の時は２枚のペアをバラバラに出してもいい
	       			if(cardNum(hand[i].getNumber(), hand) == 2 && pile.size() == 1 && hand.size() <= 5)
		 			break;
	     		}
	   	} 
		//jokerを持っている時に２枚出しで出すことで親をとって次のターンに上がる
		//ことができるならjokerとその次に強いカードをペアとして出す
	  	if( (cardNum(hand[0].getNumber(), hand) < hand.size()) && hand[cardNum(hand[0].getNumber(), hand)].equals(hand[hand.size()-2])
	  			&& checkSt2(hand[hand.size()-2].getNumber())
				&& hand[hand.size()-1].isJoker()
				&& pile.size() == 2){
	    		c2 = hand.size()-2;
	    		for(int j = 0; j < 2; j++){
	      			inHand().pickup(tmp, c2); 
	      			s.insert(tmp);
	    		}
	    		return true;
	  	}
	  	
		//場が1枚出しの時に手札の最強カードを出して親になれてかつ次のターンで上がれるならば最強カードを出す
		num = cardNum(hand[0].getNumber(), hand);  
		if( num < hand.size() &&
				cardEquals(hand[num], hand[hand.size()-1]) && checkSt(hand[hand.size()-1].getNumber()) && pile.size() == 1 && cardGreaterThan(hand[hand.size()-1], tmp2))
	    		i = hand.size()-1;
			
		//出せるカードがないとき
	  	if(i >= hand.size()){
#ifndef TAIKAI_HONBAN
	  		printf("俺の必殺技くらえ!");
#endif
	     		return true;
	   	}
		   
		//手札が4枚以上の時に出せるカードがjokerしかなく、手札に1と2がない場合で場のカードが13以下ならパス
	  	if(hand[i].isJoker() && !(cardNum(1, hand) >= 1 || cardNum(2, hand) >= 1) && !(pile[0].getNumber()+10 % 13 >= 11) && hand.size() > 4)
		    	return true;
			
		//2は1枚出しの時だけ使う
	  	if(hand[i].getNumber() == 2 && pile.size() >= 2)
		    	return true;
		
		//手札が4枚以上で最強カードを選んでいたら確実に親が取れない限りパス
	   	if(hand[i].equals(hand[hand.size()-1]) && !checkSt(hand[i].getNumber()) && hand.size() > 3){
#ifndef TAIKAI_HONBAN
		     	printf("俺はあえて出さない!");
#endif
		     	return true;
	   	}
		//
		if(cardEquals(hand[i], hand[hand.size()-1]) && !checkSt(hand[i].getNumber()) && hand.size() > 3){
#ifndef TAIKAI_HONBAN
		       	printf("俺はあえて出さないぜ!");
#endif
		       	return true;
	   	}

		//2番めに強いカードは手札が5枚以下になるまで温存。但し右から2番めと指定しているので最強カードと同じ数の場合はある。
	   	if(hand[hand.size()-2].equals(hand[i]) && hand.size() >= 6)
		     	return true;
		
		
		//手札が3枚か4枚より多いとき(ここはランダムに決まる)に最強カードを選んでいて、最強カードの次に強いのが8以下の場合はパス
	   	if(hand[i].equals(hand[hand.size()-1]) && hand[i-1].getNumber() < 9 && hand.size() > getRandom(3,4)){
			/*この下の改めて見たらなんか意味がわからん。意味があったかもしれんけど思い出せん。多分ない方がいい
			if(memory.size() > 40){
	       		flag = 0;
	       		for(c2 = 0; c2 < hand.size(); c2++){
		 			if(cardNum(hand[c2].getNumber(), hand) >= 3)
		   				flag = 1;
	       		}
	       		if(flag == 0)
		 			return true;
	     	}*/ 
	     	//else{
	       		return true;
	     	//}
			
	   	}
		 
	   	for(int j = 0; j < pile.size(); j++){
		     	inHand().pickup(tmp, i); 
		     	s.insert(tmp);
	   	}
	}

	else{// 親
	  	if( cardNum(hand[0].getNumber(), hand) < hand.size() &&
	  			hand[cardNum(hand[0].getNumber(), hand)].equals(hand[hand.size()-1]) && checkSt(hand[hand.size()-1].getNumber())){//自分の最強カードを使うことで親がまたとれて、次のターンで上がれるならば最強カードを出す
		    	inHand().pickup(tmp, hand.size()-1); 
		     	s.insert(tmp);
		     	return true;
	  	}
	  	else if( cardNum(hand[0].getNumber(), hand) < hand.size()
	  			&&hand[cardNum(hand[0].getNumber(), hand)].equals(hand[hand.size()-2]) && checkSt2(hand[hand.size()-2].getNumber()) && cardEquals(hand[hand.size()-2], hand[hand.size()-1])){//上のやつの最強カードが2枚ペアである場合
		    	c = 2;
		    	c2 = hand.size()-2;
		    	for(int j = 0; j < c; j++){
		      		inHand().pickup(tmp, c2); 
		      		s.insert(tmp);
		    	}
		    	return true;
	  	}
		else{//その他の場合
		      	flag = 0;
			//なるべく多くのカードを出せる数を探す最初のfor文で枚数を次のfor文で手札のポインタを表す。
			//出すカードを決めたらbreakでループを抜ける、但し選んだカードが11以上なら温存するため選ばない
		      	for(c = 4; c >= 1; c--){
				for(c2 = 0; c2 < hand.size(); c2++){
		  			if(cardNum(hand[c2].getNumber(), hand) == c && (hand[c2].getNumber()+10) % 13 < 8){
			    			flag = 1;
			    			break;
		  			}	
				}
				if(flag == 1)
		  			break;
	      		}
			//もし手札に11以上しか無い場合はその中で1番弱いカードを枚数分出す
	      		if(c == 0){
				c2 = 0;
				c = cardNum(hand[c2].getNumber(), hand);
	      		}
			//上で決めたカードを決めた枚数出す
	    		for(int j = 0; j < c; j++){
	      			inHand().pickup(tmp, c2); 
	      			s.insert(tmp);
	    		}
	  	}
	}  
	return true;
}

/*
 * ソートに使う順序関数の例．
 * 自分のならべたい順序の定義を作成する．
 * これはカード同士の比較であり，カードの集合同士の比較ではない．
 */
bool Group7::cardLessThan(const Card & c1, const Card & c2) {
	if ( c1.getNumber() < c2.getNumber() )
		return true;
	return false;
}

/*
 * 順序関数と整合性のある同値（等しい）関係．
 */
bool Group7::cardEquals(const Card & c1, const Card & c2) {
	return !cardLessThan(c1, c2) && !cardLessThan(c2, c1);
}

/*
 * 順序関係 compareCards を使うナイーヴな naive ソートの例．
 * 枚数は少ないので，効率は気にしない．
 */
void Group7::sortInHand(void) {
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

//手札を左が弱いカードになるような順番に並び替える
void Group7::sort(void) {
	for(int i = 0; i+1 < hand.size(); i++) {
		for(int j = i+1; j < hand.size(); j++) {
			if (cardGreaterThan(hand[i],hand[j])) {
				Card t = hand[i];
				hand[i] = hand[j];
				hand[j] = t;
			}
		}
	}
}

//memoryを弱いものが最初に来るように並び替える
void Group7::memorysort(void) {
	for(int i = 0; i+1 < memory.size(); i++) {
		for(int j = i+1; j < memory.size(); j++) {
		  if (cardGreaterThan(memory[i],memory[j])) {
				Card t = memory[i];
				memory[i] = memory[j];
				memory[j] = t;
			}
		}
	}
}

//引数のカードセットの中から引数の数が何枚あるかを調べる。
int Group7::cardNum(int num, CardSet tmp){
	int c = 0;
	for(int i = 0; i < tmp.size(); i++){
    	if(tmp[i].getNumber() == num)
      		c++;
  	}
  	return c;
}

//今までに引数の数が何枚使われているかを調べる。
int Group7::checkNum(int num){
  return cardNum(num, memory);
}

//引数の数より強いカードが使われているかを調べる（一枚出しの時）。
//jokerが出てなかったらfalse,指定のカードより強いカードが4枚memoryになかったらfalse
//trueになったならば引数の数のカードを使うことで確実に親が取れる
bool Group7::checkSt(int num){
	if(!memory[memory.size()-1].isJoker() && !hand[hand.size()-1].isJoker())
    	return false;
  	else if(hand[hand.size()-1].isJoker())
    	return true;
    for(int i = ((num + 10) % 13 )+1; i < 13; i++){
    	int cards = 0;
    	if(i != 10)
      		cards  = checkNum((i+3)%13) + cardNum((i+3)%13, hand);
    	else
        	cards  = cardNum(13, memory) + cardNum(13, hand);
    	if(cards != 4)
      		return false;    
  	}
  	return true;
}

int Group7::getRandom(int min, int max){
  	return min + (int)(rand()*(max-min+1.0)/(1.0 + RAND_MAX));
}

//上のやつの2枚出しの時で調べるバージョン
//jokerが出てない場合は調べることが出来ない
bool Group7::checkSt2(int num){
	if(!memory[memory.size()-1].isJoker() && !hand[hand.size()-1].isJoker())
		return false;
  	else if(hand[hand.size()-1].isJoker())
    	return true;
    for(int i = ((num + 10) % 13 )+1; i < 13; i++){
    	int cards = 0;
    	if(i != 10)
      		cards  = checkNum((i+3)%13) + cardNum((i+3)%13, hand);
    	else
        	cards  = cardNum(13, memory) + cardNum(13, hand);
    	if(cards < 3)
    		return false;
  	}
  	return true;
}

/*
 *  Group4.cpp
 *  PlayingCard
 *
 *  Created by 下薗 真一 on 09/04/12.
 *  Modified by Kazutaka Shimada on 09/04/21.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
/*
 g++ Card.cpp  CardSet_remove.cpp SamplePlayer.cpp CardSet_shuffle.cpp CardSet.cpp Dealer.cpp main.cpp CardSet_insert.cpp CardSet_pickup.cpp Player.cpp Group4.cpp
 */

/*
 強くなるためのアイデア
 ーだせるカードの中で最弱のものをだす
 ー複数枚出しの実装
 ー場がn枚出しの時は手札にn枚のものしか出さない
 ージョーカー，2が4枚が出ているかの判定
 ー2はなるべく1枚出し（特にジョーカーが出た場合）
 ージョーカーはなるべく残しておく　なるべく1枚出し
 ー序盤は1枚出しで1枚のものを消費する．
 ー終盤は複数枚出し中心．
 
 ー最後2枚の時，最強のカードを持ってる時は最強のカードをだす．
 */
#include <iostream>
#include <string>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"

#include "Group4.h"

void Group4::ready() {
    // 最初にカードを配られた状態
    mymemory.makeEmpty(); //memory.clear();
    sort();
    
}

//出すカードを選択
bool Group4::follow(const GameStatus & gstat, CardSet & s) {
    
    CardSet pile(gstat.pile);   //場に出ているカード
    CardSet mycards;            //自分が出すカード
    
    Card card;
    //場の状態を表示//////////////////////////////////////////////////////////////////////表示
//    std::cout << std::endl << gstat << std::flush;
    //残りカード合計
    //std::cout << "残りカード: " <<numExitCards() <<std::endl;
    //昇順ソート
    sort();
    //手札の平均
    //std::cout << "平均: " << average(hand) << std::endl;
    //手札の表示
//    std::cout << inHand() << std::endl;
    
    //std::cout << "最強：" <<mostCardStrength() <<std::endl;
    
    
    //手札が何枚あるか
    cardSetOfSameRanks(hand);
    //場札のそれぞれのカードが何枚あるか
    //pileOfSameRanks(mymemory);
    
    //カード出した人を出力
    
    /////////////////////////////////////////////////////////////////////////////////////
    
    //場にあるカードが手札の平均以下のとき
    //if ( cardStrength(pile) < average(hand) ) {
    //自分のターンになったら積極的に複数枚出しする
//    std::cout << "場札の長さ: " << pile.size() << std::endl;
    //もし自分の次の人がだしてみんなパス
    int nextperson;
    if(gstat.turnIndex== 4){
        nextperson = 0;
    }else{
        nextperson = gstat.turnIndex+1;
    }
    
//    std::cout <<pilePerson<<"," <<nextperson<<","<<gstat.numPlayers<< std::endl;
    //
    if(pilePerson == nextperson && gstat.numPlayers>2){
        findSmallestAcceptable(pile, mycards);
        hand.removeAll(mycards);
        s.insertAll(mycards);
    }
    
    
    //手札が2枚のとき
    if(hand.size()==2){
        //lastTwoCards(pile, mycards);
    }
    //俺のターン
    if(pile.size() == 0){
        
        //3~13に対して
        for(int i=3; i<=13; i++){
            //複数枚出しできるものをだす
            if(numOfEachRank[i] > 1){
                for(int j = 0; j<hand.size();j++){
                    if(i == hand[j].getNumber()){
                        //mycards.insert(hand[j]);
                    }
                }
                //hand.removeAll(mycards);
                //s.insertAll(mycards);
                //return true;
            }
        }
        //複数枚だしがなかったら
        if(mycards.size() == 0){
            //手札のなかで出せる最小のものをだす
            findSmallestAcceptable(pile, mycards);
            hand.removeAll(mycards);
            s.insertAll(mycards);
            return true;
        }
    }
    //キングまでは最小のものを出す
    /*if(pile.size() == 0){
     findSmallestAcceptable(pile, mycards);
     }*/
    else if ( cardStrength(pile) < 13) {
        //std::cout << "Hummm..." << std::endl;
        //手札のなかで出せる最小のものをだす
        findSmallestAcceptable(pile, mycards);
        //出すカードが1,2とJkrのとき
        if(13 < cardStrength(mycards)){
            if(mycards.size() > 7){
                return true;
            }else{
                hand.removeAll(mycards);
                s.insertAll(mycards);
                return true;
            }
        }
    }
    //1,2,JKrが場にでているとき
    else {
        //ここの処理は重要
        //2枚出しはしない
        if(pile.size() >= 2){
            return true;
        }
        
        //手札のなかで出せる最小のものをだす
        if(hand.size()<7){
            findSmallestAcceptable(pile, mycards);
            if ( mycards.size() > 0 ) {
                hand.removeAll(mycards);
                s.insertAll(mycards);
                return true;
            }
        }
    }
    
    if ( mycards.size() > 0 ) {
        hand.removeAll(mycards);
        s.insertAll(mycards);
        return true;
    }
    // the card identical to tmp is already removed from the hand.
    // cardSetOfSameRanks(s, pile.size());
    // たとえば、複数枚のカードを探す関数。ただしこの関数は未実装。
    // 現状ではこの follow は Player.cpp のものと等価
    return true;
}

//場を見る関数
bool Group4::approve(const GameStatus & gstat) {
//    std::cout << gstat.pile << std::endl;
    if(gstat.pile.size()>0){
        mymemory.insert(gstat.pile);
        pilePerson = gstat.turnIndex;
    }
//    std::cout << "だしたやつ:" << pilePerson << std::endl;
    //今まで出たカード
    //std::cout << "過去に出たカード一覧:" << mymemory << std::endl;
    
    return true;
}

/*
 * 順序関係 compareCards を使うナイーヴ naive なソート．
 * 枚数は少ないので，効率は気にしない．
 */
void Group4::sort(bool ascending) {
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
//　　　　　　　　　　　　　　　　　　　　　　　　自分　　　　　　　　　　　　場にあるカード
bool Group4::cardsStrongerThan(const CardSet & left, const CardSet & right) {
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

//カードの強さ
int Group4::cardStrength(const CardSet & cs) {
    int i;
    //カードが空なら0
    if ( cs.isEmpty() )
        return 0;
    //カードが1枚でジョーカーなら．．．
    if ( cs.size() == 1 && cs[0].isJoker() ) {
        return cardStrength(cs[0]);
    }
    //全てのカードに対して
    for (i = 0; i < cs.size(); i++) {
        if (!cs[i].isJoker()) {
            break;
        }
    }
    //再帰
    return cardStrength(cs[i]);
}

//その数字が手札の中で1枚だけ:true　0,2枚以上:false
bool Group4::checkRankUniqueness(const CardSet & cs) {
    //
    int rank = 0;
    
    if (cs.size() == 0)
        return false;
    //
    if ( cs.size() == 1 && cs[0].isJoker() )
        return true;
    //全てのカードに対して
    for (int i = 0; i < cs.size(); i++) {
        if (cs[i].isJoker() )
            continue;  // Jkrをスキップ
        //rankが0ならカードの番号をrankにする
        if ( rank == 0 ) {
            rank = cs[i].getNumber();
        }//もrank
        else if ( rank != cs[i].getNumber() ) {
            return false;
        }
    }
    return true;
}

//平均
double Group4::average(const CardSet & cset) {
    double sum = 0;
    if ( cset.size() == 0 )
        return sum;
    for(int i = 0; i < cset.size(); i++) {
        sum += cardStrength(cset[i]);
    }
    return sum/cset.size();
}

//カードの強さ
int Group4::cardStrength(const Card & c) {
    if ( c.isJoker() )
        return 18;      //JKrは強い
    if ( c.getNumber() <= 2 )   //1,2に対して
        return c.getNumber() + 13;
    return c.getNumber();
}

//出せるカードの中で最小のものを見つける
CardSet & Group4::findSmallestAcceptable(const CardSet & cs, CardSet & mycs) {
    // assumes the hand is sorted in ascending order
    mycs.makeEmpty();
    int cssize = cs.size();                 //場に出ているカードの枚数
    int csstrength = cardStrength(cs);      //場にあるカードの強さ
    int i, n;
    //複数出し
//    if ( cssize > 1 )
//        std::cout << "multiple cards!!!" << std::endl;
    //手札に対して
    for(i = 0; i < hand.size(); ) {
        //場にあるカードの方が強いときは次のカードへ
        if ( cardStrength(hand[i]) <= csstrength ) {
            ++i;
            continue;
        }
        //場にあるカードより強くなった手札に対して
        for(n = 1; i + n < hand.size(); ) {
            //手札がジョーカーまたは，
            if (cardStrength(hand[i + n]) == cardStrength(hand[i]) ) {
                ++n;
                continue;
            }
            break;
        }
        //手札が1,2なら1枚出し
        if(cardStrength(hand[i]) > 13){
            mycs.insert(hand[i]);
            
            return mycs;
        }
        //手札が絵札の時
        if(11 <= cardStrength(hand[i]) && cardStrength(hand[i])<=13){
            //過去に26枚以上出されている．
            if(26<mymemory.size()){
                mycs.insert(hand[i]);
                return mycs;
            }
        }
        //複数枚あって場は1枚の時
        if(n>1 && cssize == 1){
            i += n;
            continue;
        }
        
        //場にあるカードが出せる手札よりと同じ長さのとき
        if ( cssize <= n ) {
            if ( cssize != 0 )
                n = cssize;
            //出すカードに追加する
            for(int j = i; j < i + n; j++)
                mycs.insert(hand[j]);
            return mycs;
        }

        i += n;
    }
    return mycs;   // empty card set.
}

//2枚で出せるカードは1枚で出さない

// the card identical to tmp is already removed from the hand.
// cardSetOfSameRanks(s, pile.size());
// たとえば、複数枚のカードを探す関数。ただしこの関数は未実装。
// 現状ではこの follow は Player.cpp のものと等価
void Group4::cardSetOfSameRanks(CardSet &hand){
    
    //初期化
    for(int i=0;i<13;i++){
        numOfEachRank[i]=0;
    }
    //
    for(int i=0;i<hand.size();i++){
        //手札がジョーカーのときは
        if ( hand[i].isJoker()){
            numOfEachRank[0]++;
            continue;
        }
        //手札のそれぞれの数字が何枚あるかカウント
        numOfEachRank[hand[i].getNumber()]++;
    }
    //表示
//    for(int i=0;i<=13;i++){
//        std::cout << i << " " << numOfEachRank[i] << " ";
//    }
}

//一番強いカードを調べる
int Group4::mostCardStrength(){
    int count[14];
    //初期化
    for (int i = 0; i <14 ;i++){
        count[i] = 0;
    }
    //場に出ているカードに対して
    
    std::cout << "場札長さ:"<<mymemory.size() << std::endl;
    for (int i=0;i<mymemory.size();i++){
        //Jkrなければ0
        if( mymemory[i].isJoker()){
            count[0]++;
        }
        
        count[mymemory[i].getNumber()]++;
    }
    
//    for (int i = 0 ; i <14 ;i++){
//        std::cout << i << ":"<<count[i] << std::endl;
//    }
    //ジョーカー出ていない
    if(count[0]==0){
        return 18;
    }
    //2が4枚出ていない
    if(count[2]!=4){
        return 15;
    }
    if(count[1]!=4){
        return 14;
    }
    for(int i=13;i>=3;i++){
        if(count[i]!=4){
            return i;
        }
    }
    
    return 0;
}

//最後2枚の時
CardSet & Group4::lastTwoCards(const CardSet & cs, CardSet & mycs) {
    mycs.makeEmpty();
//    int cssize = cs.size();                 //場に出ているカードの枚数
    int csstrength = cardStrength(cs);      //場にあるカードの強さ
 //   int i, n;
    
    Card weak,strong;       //2枚のカードの強弱
    std::cout << hand[0] <<hand[1] ;
    if(cardStrength(hand[0])==cardStrength(hand[1])){
        mycs.insert(hand[0]);
        mycs.insert(hand[1]);
    }
    else if(cardStrength(hand[0])<cardStrength(hand[1])){
        weak = hand[0];
        strong = hand[1];
    }else{
        weak = hand[1];
        strong = hand[0];
    }
    
    //手札に対して
    //場にあるカードは最強ならだす
    if (cardStrength(strong) == mostCardStrength()) {
        mycs.insert(strong);
        return mycs;
    }
    //場にあるカードより弱いカードをは積極的に出す
    if (csstrength < cardStrength(weak)) {
        mycs.insert(weak);
        return mycs;
    }
    if (csstrength < cardStrength(strong)) {
        mycs.insert(strong);
        return mycs;
    }
    
    return mycs;   // empty card set.
}

















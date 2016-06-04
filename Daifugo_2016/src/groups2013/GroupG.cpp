/*
 *  LittleThinkPlayer.cpp
 *  PlayingCard
 *
 *  Created by 下薗 真一 on 09/04/12.
 *  Modified by Kazutaka Shimada on 09/04/21.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include <iostream>
#include <string>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"
#include "GroupG.h"
using namespace grp2013;

/*
 * コンストラクタ
 */
GroupG::GroupG(const char * s) : Player(s) {
	this->myCardSize = 0;
}


/*
 * 出されたカードを確認する関数
 * CardSet &pile: 出されたカード
 * int numCards[]: 各プレイヤーの手札数
 *
 * @return bool: 成功時:true, 失敗時:false
 */
bool GroupG::approve(const CardSet & pile, int numCards[]) {

	// new-game flag
	if (inHand().size() > this->myCardSize) {
		this->memory.makeEmpty();
		this->restCards.setupDeck();
	}

	this->rivalCardSize = numCards;
	this->myCardSize = inHand().size();

	// 出されたカード軍を記憶しておく
	this->memory.insert(pile);

	// 余りカードデータから出たカードを消す
	this->removeCards(this->restCards, pile);
	this->removeCards(this->restCards, inHand());

	printf("-");
	pile.print();

/*
	int myId = getId();

	for (int i=0; numCards[i] != NO_MORE_PLAYERS; i++) {
		if (i == myId)
			printf("あなた : %d 枚\n", i, numCards[i]);
		else
			printf("player %d : %d 枚\n", i, numCards[i]);
	}
*/
	return true;
}


/*
 * 最適なカードを計算して出す関数
 * @CardSet &pile: 出されたカード。自分が始めのときは空となる。
 * @CardSet &s: カードを出す用の箱
 *
 * @return bool: 成功時:true, 失敗時:false
 */
bool GroupG::follow(const CardSet &pile, CardSet &s) {
	int locate = -1;
	int sendSize = 0;

	// sort & print
	this->sort(this->memory);
	this->sort(this->restCards);
	this->sort(this->inHand());
	//this->printDetail();

	// 思考
	this->think(pile, locate, sendSize);

	// カードを出す
	this->sendCard(locate, sendSize, s);

	return true;
}


/*
 * 複数のカードを削除する関数
 * @CardSet cards: カードセット
 * @CardSet removeCards: 削除するカードセット
 */
void GroupG::removeCards(CardSet &cards, CardSet removeCards) {
	int size = removeCards.size();
	while ( --size > -1 ) {
		cards.remove(removeCards.at(size));
	}
}



/*
 * 弱い順（3,4,5,...,K,1,2,0）にソートする関数
 * @CardSet &cards: ソートするカード
 */
void GroupG::sort(CardSet &cards) {
	int size = cards.size();
	int cardData[size][2];
	int i, j;
	int tmpData[2];
	Card tmpCard;

	// cardDataに保存
	for ( i = 0; i < size; i++) {
		cardData[i][0] = cards.at(i).getRank();
		cardData[i][1] = cards.at(i).getSuit();
	}

	// cardDataをBubbleSort
	for ( i = 0; i < size - 1; i++) {
		for ( j = size - 1 ; j > i; j--) {
			if ( getRevisedRank(cardData[j][0]) < getRevisedRank(cardData[j - 1][0]) ) {
				tmpData[0] = cardData[j][0];
				tmpData[1] = cardData[j][1];
				cardData[j][0] = cardData[j - 1][0];
				cardData[j][1] = cardData[j - 1][1];
				cardData[j - 1][0] = tmpData[0];
				cardData[j - 1][1] = tmpData[1];
			}
		}
	}

	// cardDataをcardsに直す
	cards.makeEmpty();
	for ( i = 0 ; i < size; i++) {
		tmpCard.set( cardData[i][1] , cardData[i][0] );
		cards.insert(tmpCard);
	}
}

/*
 * 状況をprintする
 */
void GroupG::printDetail() {
	printf("\n***********************************\n");
	printf("memory : %d枚 ", this->memory.size());
	this->memory.print();
	printf("rest : %d枚 ", this->restCards.size());
	this->restCards.print();
	printf("myDeck : %d枚 ", this->inHand().size());
	this->inHand().print();
	printf("***********************************\n");
}


/*
 * 出すカードを考える
 * @CardSet pile: 出されたカード軍
 * @int &locate: 出すカードの位置
 * @int &sendSize: 出すカードの枚数
 *
 * @int locate: 出すカードの位置
 */
void GroupG::think(CardSet pile, int &locate, int &sendSize) {
	bool bFirstSet = pile.size() == 0 ? true : false;
	int minSize = this->getMinCardSize();
	//int rank;

	// なんでも
	if ( bFirstSet ) {

		// 最弱カードが10未満
		if (this->getRevisedRank(this->inHand().at(0).getRank()) < 10) {
			locate = 0;
			sendSize = this->getSizeOfSameRankCard(this->inHand(), inHand().at(locate).getRank());

		// 相手のカードが3枚以下
		} else if (minSize <= 3) {
			// できるだけ複数枚
			for (int i = 0; i < inHand().size(); i++) {
				sendSize = this->getSizeOfSameRankCard(this->inHand(), inHand().at(i).getRank());
				if (sendSize >= 2) {
					locate = i;
					break;
				}
			}
		} else {
			// できるだけ１枚
			for (int i = 0; i < inHand().size(); i++) {
				sendSize = this->getSizeOfSameRankCard(this->inHand(), inHand().at(i).getRank());
				if (sendSize == 1) {
					locate = i;
					break;
				}
			}
		}

		if (locate == -1) {
			locate = 0;
			sendSize = this->getSizeOfSameRankCard(this->inHand(), inHand().at(locate).getRank());
		}

	// 相手より強いカードを出す
	} else {
		// 相手のカードが13未満だったらとりあえず出す
		if (this->getRevisedRank(pile.at(0).getRank()) < 13) {
			locate = this->getGreater( pile.at(0), pile.size() );
			sendSize = pile.size();
		// 相手のカードが強かったら
		} else {
			if ( minSize <= 4 ) {
				if (pile.size() == 1) {
					locate = this->getGreater( pile.at(0), pile.size() );
				} else {
					if (inHand().size() > pile.size()) {
						locate = this->getGreater( pile.at(0), pile.size() );
					}
				}
				sendSize = pile.size();
			}

			// 相手のカードの最強カードを見る
			//if (this->restCards.at(this->restCards.size() - 1) > this->inHand().at(locate)) {
			//	locate = -1;
			//}
		}
	}

	// 手札組み合わせ
	//int restSize = restCards.size();
	//int rivalSize = rivalCardSize[0];

}


/*
 * 相手が持っているであろう組み合わせを出す
 */
void GroupG::setRankCombination(int k, int n, int r) {


}


/*
 * ランクを強さによって補正する（補正値は適当）
 * @Card rank: 補正するランク
 * @int bAppend: 補正の付加/除去
 */
int GroupG::getRevisedRank(int rank) {

	if(rank == 1 || rank == 2) {
		rank += 20;
	} else if (rank == 0) {
		rank += 40;
	}

	return rank;
}


/*
 * ターゲットより大きなカードの"位置"を取得する関数
 * @Card target: ターゲット
 * @int sendSize: 出すカードの枚数
 *
 * @return int: ターゲットより大きいカードの位置
 */
int GroupG::getGreater(Card target, int sendSize) {
	int locate = -1;
	int size = this->inHand().size();
	int handRank;
	int targetRank;
	int sizeOfSameCard;
	Card tmpCard;

	// 既に弱い順にソートされているので、位置が小さい順に探す。
	for (int i = 0; i < size; i++) {
		handRank = this->inHand().at(i).getRank();
		targetRank = target.getRank();

		if ( getRevisedRank(handRank) > getRevisedRank(targetRank) ) {
			sizeOfSameCard = this->getSizeOfSameRankCard(inHand(), handRank);

			if( sizeOfSameCard >= sendSize ) {
				locate = i;
				break;
			}
		}
	}

	return locate;
}

/*
 * 任意のランクと同じランクのものが何枚あるかを出す関数
 * @CardSet target: 対象
 * @int targetRank: 探すランク
 *
 * @return int: 同じランクのものの枚数
 */
int GroupG::getSizeOfSameRankCard(CardSet cards, int targetRank) {
	int count = 0;
	int size = cards.size();

	for ( int i = 0; i < size; i++ ) {
		if( targetRank == cards.at(i).getRank()) {
			count++;
		}
	}
	
	return count;
}


int GroupG::getMinCardSize() {
	int min = 100;
	for (int i = 0; this->rivalCardSize[i + 1] != GameStatus::NO_MORE_PLAYERS; i++ ) {
		if (rivalCardSize[i] != 0 && rivalCardSize[i] < min) {
			min = rivalCardSize[i];
		}
	}
	return min;
}


/*
 * カードを出す関数
 * @int locate: 同番号の位置
 * @int sendSize: 出すカードの枚数
 * @CardSet s: 出すカードを入れる箱
 */
void GroupG::sendCard(int locate, int sendSize, CardSet &s) {
	Card card;
	if(locate != -1) {
		s.makeEmpty();
		for ( int i = 0; i < sendSize; i++ ) {
			this->inHand().pickup(card, locate);
			s.insert(card);
		}
	}
}


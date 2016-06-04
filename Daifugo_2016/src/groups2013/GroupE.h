/*
 * GroupEPlayer1.h
 * Created by 一瀬航 on 13/05/02
 *
 */
#include "Player.h"

namespace grp2013 {

class GroupE : public Player {
  CardSet memory;
  int singlcard[13];
  int twocards[13];//添字0~12の配列の中に1~13のカードで2ペアのものに1、それ以外は0
  int threecards[13];
  int fourcards[13];
  int two_flag;
  int three_flag;
  int four_flag;
  int joker_flag;

 public:
  GroupE(const char *);
  bool follow(const CardSet &, CardSet &);
  bool approve(const CardSet &,const int[]);
  Card min_card(Card c[],int limit);
  void SearchCards(void);
  CardSet PickUpCard(int i);

	bool follow(const GameStatus & gstat, CardSet & cards) {
	  return follow(gstat.pile, cards);
	}
	bool approve(const GameStatus & gstat) {
	  return approve(gstat.pile, (int *) gstat.numCards);
	}


 };

}

#include "Player.h"

namespace grp2015 {


class Group5 : public Player{
  CardSet memory;
  CardSet trump;

public:
 Group5(const char * name = "n231054s") : Player(name){
    //初期化
}
  virtual void ready();
  virtual bool follow(const GameStatus &, CardSet &);
  virtual bool approve(const GameStatus &);
 
  //独自の関数群
  int getPosbyPile(CardSet & pile);
  int insert(int pilenum);
  
        bool cardLessThan(const Card & c1, const Card & c2);
	bool cardEquals(const Card & c1, const Card & c2);
	void sortInHand();
	void sortInTrump();
	bool is2PareinHand(CardSet & s,int & pos);
	bool is3PareinHand(CardSet & s,int & pos);
	bool is4PareinHand(CardSet & s, int & pos);
};

}

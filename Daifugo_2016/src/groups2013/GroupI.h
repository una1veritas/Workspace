
#include "Player.h"

namespace grp2013 {
class GroupI : public Player {
  CardSet memory;

public:
	GroupI(const char *);
	bool follow(CardSet &, CardSet &);
	bool approve(CardSet &, int[]);
	bool sort(CardSet&);
	bool cardSetOfSameRanks(CardSet & s, int size);
	bool cardSetOfSameRanks1(CardSet & s, int size);
	
};
}

/*
 *  Dealer.h
 *  PlayingCard
 *
 *  Created by 下薗 真一 on 09/04/12.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef _DEALER_H_
#define _DEALER_H_

#include <vector>
#include <map>
#include <cstdlib>
#include <ctime>

#include "Player.h"
#include "GameStatus.h"

class Dealer {
public:
	const static int NUM_OF_ALL_CARDS = 53;
	const static int NUM_OF_MAX_PLAYERS = 10;

private:
	CardSet theDeck;
	CardSet discarded;
	bool turnPassed;

	int numParticipants;
	struct {
		unsigned long id;
		Player * player;
		CardSet shadow;
	} participant[NUM_OF_MAX_PLAYERS];

	struct {
		float point;
		std::map<int, int> places;
	} stats[NUM_OF_MAX_PLAYERS];

	std::vector<int> playOrder;
	std::vector<int> finishedOrder;
	int turnIndex;
	int leaderIndex;

	// Rule flags
	bool noMillionaire;


public:
	Dealer(void);
	~Dealer();

	static bool checkRankUniqueness(const CardSet &);
	static int getCardStrength(const CardSet &);

	void newGame(bool changeOrder = false);
	bool regist(Player *);
	bool deal();
	void show();
	
	int howManyPlayingPlayers() const;
	int howManyParticipants() const;
	int howManyFinishedPlayers() const;

	void showDiscardedToPlayers();
//	void chooseLeader();
//	void promptNext(CardSet & t);
	
	bool acceptCheck(const CardSet & opened) const;
	void accept( CardSet & opened);
	void reject( CardSet & opened);

//	void replaceWith(CardSet &, CardSet &);
	const CardSet & discardPile();
	void clearDiscarded();
	bool playerInTurnIsLeader() const;
	void setAsLeader(void);
	void setAsLeader(const int ith);

	void withdrawPlayerInTurn(void);

	int playerID(const Player & p) const;
	const Player & player(const int id) const;
	Player & player(const int id);
	
	Player & playerInTurn();
	Player & nextPlayer();
	const Player & currentLeader() const;
	Player & finishPlayerInTurn();
	
	GameStatus gameStatus(void) const;

	Player & finishedAt(int i);
	float & finishPoint(const Player & p);
	std::map<int, int> & places(const Player & p);
	void setPlayOrderByFinishedRank(void);
	void givePointsByFinishedRank(void);
	void sortFinishedPlayersByPoint(void);

private:
	void shuffleOrder(void);

};

#endif

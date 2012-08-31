/*
 *  Dealer.h
 *  PlayingCard
 *
 *  Created by 下薗 真一 on 09/04/12.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

class Dealer {
	CardSet theDeck;
	CardSet discarded;
	
	Player * players[10];
	int numberOfPlayers;
	int pauper;
	int turn;
	int leaderIndex;
	
//	Player * ranking[10];
	bool noMillionaire;
public:
	Dealer();

	void newGame();
	bool registerPlayer(Player *);
	bool deal(int);
	bool dealAll();
	void letemShow();
	
	int howManyPlayers() { return numberOfPlayers; }
	int howManyParticipants();
		
	void hailThePlayers() { return; }
	void showDiscardedAround();
	void chooseLeader();
	
	void promptNext(CardSet & t);
	
	bool accept(CardSet & opened);
	bool checkRankUniqueness(CardSet &);

//	void replaceWith(CardSet &, CardSet &);
	CardSet & discardPile();
	void clearDiscardPile();
	bool playerInTurnIsLeader();
	void setAsLeader();
	Player & player(int);
	int numberOfFinishedPlayers() ;
	void withdrawPlayer(int);
	
	Player & playerInTurnFinished();
	
	Player & playerInTurn();
	Player & nextPlayer();
	Player & currentLeader();
	
};

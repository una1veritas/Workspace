/*
 *  Dealer.cpp
 *  PlayingCard
 *
 *  Created by 下薗 真一 on 09/04/12.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream> 
#include <string>

#include <cstdlib>
#include <ctime>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"
#include "Dealer.h"
#include "GameStatus.h"

// class methods.

Dealer::Dealer() {
	playOrder.clear();
	finishedOrder.clear();
	turnIndex = 0;
	turnPassed = false;
	numParticipants = 0;
	leaderIndex = 0;
	// set rules
	noMillionaire = true;
	//
	std::srand(std::time(0));
	return;
}

Dealer::~Dealer(void) {
	for(int i = 0; i < numParticipants; i++) {
		delete(participant[i].player);
	}
}

const Player & Dealer::player(const int id) const {
	return *participant[id].player;
}

Player & Dealer::player(const int id) {
	return *participant[id].player;
}

// the number of playing, not finished, players
int Dealer::howManyPlayingPlayers() const {
	return playOrder.size();
}


bool Dealer::checkRankUniqueness(const CardSet & cs) {
	int rank = 0;

	if (cs.size() == 0)
		return false;

	if ( cs.size() == 1 && cs[0].isJoker() )
		return true;

	for (int i = 0; i < cs.size(); i++) {
	  if (cs[i].isJoker() )
		  continue;  // Jkrをスキップ
	  if ( rank == 0 ) {
		  rank = cs[i].getNumber();
	  } else if ( rank != cs[i].getNumber() ) {
	    return false;
	  }
	}
	return true;
}


// instance methods.
bool Dealer::regist(Player * pl) {
	if ( !(numParticipants < NUM_OF_MAX_PLAYERS) ) {
		std::cerr << "Error: tried player registration exceeding the limit." << std::endl;
		return false;
	}
	pl->id = numParticipants;
	participant[numParticipants].id = pl->id;  // participant[i].id == i
	participant[numParticipants].player = pl;
	participant[numParticipants].shadow.clear();
//	participant[numParticipants].finished = true;
	stats[numParticipants].point = 0;
	numParticipants++;
	return true;
}

const Player & Dealer::currentLeader() const {
	return player(playOrder.at(leaderIndex) );
}

bool Dealer::playerInTurnIsLeader() const {
	return leaderIndex == turnIndex;
}

void Dealer::newGame(bool changeOrder) {
	for (int i = 0; i < numParticipants; i++) {
		//participant[i].finished = false;
		participant[i].player->clearHand();
		participant[i].player->ready();
		participant[i].shadow.clear();
	}

	playOrder.clear();
	if ( changeOrder || (finishedOrder.size() != (unsigned int) numParticipants) ) {
		for (int i = 0; i < numParticipants; i++) {
			playOrder.push_back(i);
		}
		shuffleOrder();
	} else {
		for (int i = 0; i < numParticipants; i++)
			playOrder.push_back(finishedOrder[i]);
	}
	finishedOrder.clear();
}

void Dealer::shuffleOrder(void) {
	unsigned long s, d;
	int t;
	if ( playOrder.size() != (unsigned int) numParticipants )
		return;
	for(int i = 0; i < 19; i++) {
		s = std::rand() % playOrder.size();
		d = std::rand() % playOrder.size();
		if ( s == d ) continue;
		t = playOrder[s];
		playOrder[s] = playOrder[d];
		playOrder[d] = t;
	}
	return;
}

void Dealer::setAsLeader(void) {
	leaderIndex = turnIndex;
}

void Dealer::setAsLeader(const int ith) {
	/*
	for (turnIndex = ith; turnIndex < ith + numParticipants; turnIndex++) {
		if ( participant[playOrder[turnIndex % numParticipants]].finished )
			continue;
	}
	*/
	turnIndex = ith % howManyPlayingPlayers();
	leaderIndex = turnIndex;
}

bool Dealer::deal(void) {
	Card top;

	for (int i = 0; i < howManyPlayingPlayers() ; i++) {
		player(playOrder[i]).clearHand();
	}
	theDeck.setupDeck();
	theDeck.shuffle();
	int index = 0;
	while ( !theDeck.isEmpty() ) {
		theDeck.pickup(top, 0);
		player(playOrder[ index ]).insert(top);
		index = (index + 1) % playOrder.size();
	}
	for(unsigned int i = 0; i < playOrder.size(); i++) {
		participant[playOrder[i]].shadow.clear();
		participant[playOrder[i]].shadow.insertAll(player(playOrder[i]).hand);
		/*
		std::cout << "player id " << playOrder[i] << " "
				<< participant[playOrder[i]].shadow
				<< std::endl;
		 */
	}
	for(unsigned int i = 0; i < playOrder.size(); i++)
		player(playOrder[i]).ready();

	turnIndex = 0;
	leaderIndex = 0;
	turnPassed = false;

	return true;
}

void Dealer::clearDiscarded() {
	unsigned int i;
	discarded.clear();
	for(i = 0; i < playOrder.size(); i++) {
		if ( !player(playOrder[i]).hand.equals(participant[playOrder[i]].shadow) )
			std::cout << "Player \"" << player(playOrder[i]).playerName() <<  "\", ID " << playOrder[i] << " Ikasama error!!!" << std::endl;
	}
}

const CardSet & Dealer::discardPile() {
	return discarded;
}


bool Dealer::acceptCheck(const CardSet & opened) const {
	int openedRank, discardedRank;

	if (opened.isEmpty() )
		return false;  // regard the empty open set as "pass"

	if (!checkRankUniqueness(opened))
		return false;
	if ( opened.size() >= 5 )
		return false;

	if ( discarded.isEmpty() )
		return true;

	if ( discarded.size() != opened.size() )  // the number of cards must be match. no five cards w/ Jkr allowed.
	  return false;
	
	openedRank = getCardStrength(opened);
	discardedRank = getCardStrength(discarded);

	if ( openedRank > discardedRank )
	    return true;
	return false;
}

void Dealer::accept(CardSet & opened) {
	// passed all the checks.
	discarded.clear();
	discarded.insertAll(opened);
	turnPassed = false;
	participant[playOrder[turnIndex]].shadow.removeAll(discarded);
}

void Dealer::reject(CardSet & opened) {
	if ( !opened.isEmpty() )
		playerInTurn().takeCards(opened);
	turnPassed = true;
	// leave as untouched the shadow set
}

int Dealer::getCardStrength(const CardSet & cs) {
	int i;
	if ( cs.isEmpty() )
		return 0;

	if ( cs.size() == 1 && cs[0].isJoker() ) {
		return 53;
	}
  	for (i = 0; i < cs.size(); i++) {
	  if (!cs[i].isJoker()) {
		  break;
	  }
	}
  	if ( cs[i].getNumber() < 3 )
  		return cs[i].getNumber() + 13;
	return cs[i].getNumber();
}

void Dealer::showDiscardedToPlayers() {
	GameStatus gstat = gameStatus();
	if ( turnPassed )
		gstat.pile.clear();
	for (int i = 1; i < howManyPlayingPlayers(); i++) {
		player(playOrder[(turnIndex + i) % howManyPlayingPlayers()] ).approve(gstat);
	}
	return;
}

void Dealer::withdrawPlayerInTurn(void) {
	int id;
	id = playOrder[turnIndex];
	participant[id].player->clearHand();
	participant[id].shadow.clear();
//	participant[id].finished = true;
	// shrink the circle of playing players'
	playOrder.erase(playOrder.begin()+turnIndex);
	finishedOrder.push_back(id);
	// if the last player and/or the leader has been withdrawn
	// avoid mod operation since playOrder.size() for the last person goes to zero.
	if ( (unsigned int) turnIndex == playOrder.size() )
		turnIndex = 0;
	if ( (unsigned int) leaderIndex == playOrder.size() )
		leaderIndex = 0;
	return;
}

Player & Dealer::finishPlayerInTurn() {
	Player & p = playerInTurn();
	withdrawPlayerInTurn();
	return p;
}


int Dealer::howManyParticipants() const {
	return numParticipants;
}


int Dealer::howManyFinishedPlayers() const {
	return finishedOrder.size();
}

Player & Dealer::playerInTurn(void) {
	return player(playOrder[turnIndex]);
}

Player & Dealer::nextPlayer() {
	turnIndex = (turnIndex+1) % howManyPlayingPlayers();
	return playerInTurn();
}

Player & Dealer::finishedAt(int i) {
	return *(participant[finishedOrder[i]].player);
}

void Dealer::show() {
		for (unsigned int i = 0; i < playOrder.size() ; i++) {
			if ( i == (unsigned int) leaderIndex )
				std::cout << "* ";
			else std::cout << "  ";
			std::cout << player(playOrder[i]).playerName()
					<< " (" << player(playOrder[i]).getID() << ")"
					<< "\t" << player(playOrder[i]).inHand() << std::endl;
		}
		if ( finishedOrder.size() > 0 ) {
			std::cout << "  " << "--------" << std::endl;
			for (unsigned int i = 0; i < finishedOrder.size() ; i++) {
				std::cout << "  " /* << player(finishedOrder[i]).getID() << " " */
						<< player(finishedOrder[i]).playerName() << "\t";
				std::cout << player(finishedOrder[i]).inHand() << std::endl;
			}
		}
}


GameStatus Dealer::gameStatus(void) const {
	GameStatus gstat;
	gstat.pile = discarded;
	gstat.turnIndex = turnIndex;
	gstat.leaderIndex = leaderIndex;
	gstat.numPlayers = howManyPlayingPlayers();
	for (int i = 0; i < gstat.numPlayers; i++)
		gstat.numCards[i] = player(playOrder[i]).size();
	gstat.numCards[gstat.numPlayers] = GameStatus::NO_MORE_PLAYERS; // for backward compatibility
	for (int i = 0; i < gstat.numPlayers; i++)
		gstat.playOrder[i] = playOrder[i];
	return gstat;
}

int Dealer::playerID(const Player & p) const {
	for (int id = 0; id < howManyParticipants(); id++) {
		if ( &p == participant[id].player )
			return id;
	}
	return howManyParticipants();
}

float & Dealer::finishPoint(const Player & p) {
	return stats[playerID(p)].point;
}
std::map<int, int> & Dealer::places(const Player & p) {
	return stats[playerID(p)].places;
}

void Dealer::givePointsByFinishedRank(void) {
	unsigned int i;
	for(i = 0; i < finishedOrder.size(); i++) {
		const Player & thePlayer = player(finishedOrder[i]);
		if ( i == 0 ) {
			finishPoint(thePlayer) += 0.9;
		} else {
			finishPoint(thePlayer) += i + 1;
		}
		if ( places(thePlayer).find(i) ==
				places(thePlayer).end() ) {
			places(thePlayer)[i] = 1;
		} else {
			places(thePlayer)[i]++;
		}
	}
}

void Dealer::sortFinishedPlayersByPoint(void) {
	for(unsigned int i = 0; i < finishedOrder.size() - 1; i++) {
		for(unsigned int j = i+1; j < finishedOrder.size(); j++) {
			if ( finishPoint(player(finishedOrder[i]))
					> finishPoint(player(finishedOrder[j])) ) {
				int t = finishedOrder[i];
				finishedOrder[i] = finishedOrder[j];
				finishedOrder[j] = t;
			}
		}
	}
}


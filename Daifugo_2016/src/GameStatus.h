/*
 * GameStatus.h
 *
 *  Created on: 2015/05/07
 *      Author: sin
 */

#ifndef GAMESTATUS_H_
#define GAMESTATUS_H_

#include "Card.h"
#include "CardSet.h"

#include <string>

struct GameStatus {
public:
	static const int MAXIMUM_NUM_OF_PLAYERS = 8;
	static const int NO_MORE_PLAYERS = 99;

	CardSet pile;
	int numPlayers;
	int turnIndex;
	int leaderIndex;
	int numCards[MAXIMUM_NUM_OF_PLAYERS];
	int playOrder[MAXIMUM_NUM_OF_PLAYERS];
	std::string playerName[MAXIMUM_NUM_OF_PLAYERS];
	int numParticipants;

	GameStatus(void) : pile(), numPlayers(0), turnIndex(0), leaderIndex(0), numParticipants(0) { }  // Empty instance

	std::ostream & printOn(std::ostream & out) const {
		out << "GameStatus(";
		out << "pile = " << pile;
		out << ", numPlayers = "<< numPlayers;
		out << ", turnIndex = " << turnIndex;
		out << ", leaderIndex = " << leaderIndex;
		out << ", numCards[";
		for(int i = 0; i < numPlayers; i++) {
			out << numCards[i];
			out << " ";
		}
		out << "]";
		out << ", playOrder = [";
		for(int i = 0; i < numPlayers; i++) {
			out << playOrder[i];
			out << " ";
		}
		out << "]";
		out << ", numParticipants = ";
		out << numParticipants;

		out << std::endl;
		return out;
	}

	// おまけ
	friend std::ostream & operator<<(std::ostream& ostr, const GameStatus & stat) {
		return stat.printOn(ostr);
	}

};

#endif /* GAMESTATUS_H_ */

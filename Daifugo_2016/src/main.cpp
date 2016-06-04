#include <iostream>
#include <iomanip>
#include <map>

#include <stdio.h>
#include <string.h>

#include "Card.h"
#include "CardSet.h"
#include "Dealer.h"

#include "main_players.h"


int main (int argc, char * const argv[]) {
	
	Dealer d;
	int game = 0, times = 0;
	std::string resp1, resp2;

	if ( argc >= 2 )
		times = atoi(argv[1]);
	if ( argc >= 3 ) {
		resp1 = argv[2];
	}
	if ( argc >= 4 ) {
		resp2 = argv[3];
	}
	if ( times <= 0 )
		times = 5;

	std::cout << "Registering players." << std::endl;

	registerPlayers(d);

	for (game = 0; game < times; ) {
		std::cout << std::endl << "Starting Game " << (game+1) << "." << std::endl;
		d.newGame( ((game % 100) == 0) ); // shuffle order per 100 time
		d.deal();
		bool passed = true;
		while (true) {
			while (true) {
				if (passed && d.playerInTurnIsLeader() ) {
					d.clearDiscarded();
					std::cout << std::endl << "flushed. " << std::endl;
					std::cout << "  -------- " << std::endl;
					d.show();
					std::cout << "  -------- " << std::endl;
					if ( resp1.length() == 0 )
						std::getline(std::cin, resp1);
				}
				std::cout << std::endl << d.playerInTurn().playerName() << "'s turn: ";
				GameStatus state = d.gameStatus();
				CardSet opened;
				d.playerInTurn().follow(state, opened);
				std::cout << opened << std::flush;
				if ( d.acceptCheck(opened) ) {
					std::cout << "accepted." << std::endl;
					d.accept(opened);
					passed = false;
				} else {
					if ( !opened.isEmpty() )
						std::cout << "rejected, ";
					std::cout << "pass." << std::endl;
					d.reject(opened);
					passed = true;
				}
				if (!passed) {
					d.setAsLeader();
					std::cout << "--- Lead " << d.discardPile() << " by " << d.currentLeader().playerName() << ". " << std::endl;
					// std::cin.getline(dummy,31);
				}
				std::cout << std::endl;
				d.showDiscardedToPlayers();
				if (d.playerInTurn().isEmptyHanded())
					break; // finished

				d.nextPlayer();
			}
			std::cout << d.playerInTurn().playerName() << " finished. " << std::endl << std::endl;
			d.finishPlayerInTurn();
			if (d.howManyPlayingPlayers() == 1) {
				// どべ確定
				d.finishPlayerInTurn();
				break;
			}
		}
		d.givePointsByFinishedRank();

		std::cout << std::endl 
			<< "This game's result: " << std::endl;
		for (int i = 0; i < d.howManyParticipants() ; i++) {
			std::cout << i+1 << ": "  << d.finishedAt(i).playerName() << "\t"
				/*	<< d.pointByID(d.playerID(d.finishedPlace(i))) << " " */
				 << " " << std::endl;
		}

		game++;
		if ( game < times ) {
			std::cout << std::endl << "Type to continue." << std::endl;
			if ( resp2.length() == 0 )
				std::getline(std::cin, resp2);
		}
	}

	std::cout << std::endl << "Final result: " << std::endl;
	d.sortFinishedPlayersByPoint();
	for(int i = 0; i < d.howManyParticipants(); i++) {
		std::cout << d.finishedAt(i).playerName() << ": \t"
				<< (d.finishPoint(d.finishedAt(i))/times) << " \t";
		for(int place = 0; place < d.howManyParticipants(); place++) {
			std::cout << d.places(d.finishedAt(i))[place] << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

    return 0;
}

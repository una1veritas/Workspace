#include <string>
#include <iostream>

#include <stdio.h>

#include "Card.h"
#include "CardSet.h"
#include "Player.h"
#include "LittleThinkPlayer.h"
#include "Dealer.h"

void playingLoop(Dealer & d) {
	CardSet opened;
	char dummy[32];
	bool passed;

	d.newGame();
	std::cin.getline(dummy,31);
	d.dealAll();
	d.setAsLeader();
	passed = true;
	
	while (true) {
		while (true) {
			if (passed && d.playerInTurnIsLeader() ) {
				d.clearDiscardPile();
				std::cout << "flushed. " << std::endl;
				std::cin.getline(dummy,31);
				d.letemShow();
			}
			std::cout << "Now " << d.playerInTurn().playerName() << "'s Turn: " ;
			d.playerInTurn().follow(d.discardPile(), opened);
			std::cout << opened << std::endl;
			if (opened.isempty() || !d.accept(opened)) {
				if (!opened.isempty()) {
					d.playerInTurn().takeCards(opened);
					std::cout << "Reject. ";
				}
				std::cout << "Pass." << std::endl;
				passed = true;
			} else {
				passed = false;
			}
			std::cout << std::endl;
			if (d.playerInTurn().isEmptyHanded())
				break;
			if (!passed) {
				d.setAsLeader();
				std::cout << d.discardPile().printString() 
				<< " lead by " << d.currentLeader().playerName() << ". " ;
			}
			
			d.showDiscardedAround();
			
			d.nextPlayer();
		}
		std::cout << d.playerInTurn().playerName() << " finished. " << std::endl << std::endl;
		//d.letemShow();
		d.playerInTurnFinished();
		if (d.howManyPlayers() == 1) {
			d.playerInTurnFinished();
			break;
		}
	}
	return;
}

int main (int argc, char * const argv[]) {
    // insert code here...
    std::cout << "Hello, World!\n";
	
	Dealer d;
	
	d.registerPlayer(new Player("Erika"));
	d.registerPlayer(new Player("Konomi"));
	d.registerPlayer(new Player("Sugina"));
	d.registerPlayer(new Player("Takenoko"));
	d.registerPlayer(new LittleThinkPlayer("Warabi"));
	d.hailThePlayers();

	for (int g = 1; g < 5; g++) {
		std::cout << std::endl << "Game " << g << std::endl;
		
		playingLoop(d);
		
		std::cout << std::endl 
			<< "This game's result: " << std::endl;
		for (int i = 0; i < d.howManyParticipants() ; i++) {
			std::cout << i+1;
			switch (i+1) {
					case 1:
						std::cout << "st ";
					break;
					case 2:
						std::cout << "nd ";
					break;
					case 3:
						std::cout << "rd ";
					break;
					default:
						std::cout << "th ";
					break;
			}
			std::cout << "place: " << d.player(d.howManyParticipants() - i - 1).playerName() << std::endl;
		}
	}
    return 0;
}

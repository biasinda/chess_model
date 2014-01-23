
#include "movePackage.h"



	movePackage::movePackage(){};

	movePackage::movePackage(std::vector<repast::Point<int> > newLegalMoves, int newTeam, int newOccupant): legalMoves(newLegalMoves),team(newTeam),occupant(newOccupant){};

	repast::Point<int> movePackage::getRandomMove(){

		repast::Point<int> randomMove(10,10);
		//get size of legal moves
		int moveN=legalMoves.size();


		//get a random integer between 0 and N-1
		if(moveN!=0){
		int r = rand() % moveN;

		randomMove=legalMoves.at(r);
		}

		return randomMove;
	}

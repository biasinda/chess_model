#ifndef MOVEPACKAGE
#define MOVEPACKAGE

#include "repast_hpc/Point.h"
#include <stdio.h>

typedef struct movePackage movePackage;

struct movePackage{
	int occupant;
	int team;
	std::vector<repast::Point<int> > legalMoves;

	movePackage();

	movePackage(std::vector<repast::Point<int> > newLegalMoves, int newTeam, int newOccupant);

	repast::Point<int> getRandomMove();

};




#endif

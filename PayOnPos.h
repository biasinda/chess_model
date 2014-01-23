
#ifndef PAYONPOS
#define PAYONPOS

#include "repast_hpc/Point.h"
#include <stdio.h>
#include "LastInputs.h"

typedef struct PayONPos PayONPos;

struct PayONPos{
	double payoff;
	repast::Point<int> newPosition;
	repast::Point<int> oldPosition;
	int team;
	int occupant;
	LastInputs lastInputs;

	int otherTeam;
	int occupantThatWasEaten;


	//make constuctors for the struct
	PayONPos();
	PayONPos(double _newPayoff, repast::Point<int> _positionOld,  repast::Point<int> _positionNew, int _newTeam, int _newOccupant, int _newOccupantThatWasEaten, int _newOtherTeam, LastInputs _lastInputs);

	 /* For archive packaging */
	    template<class Archive>
	    void serialize(Archive &ar, const unsigned int version){
	        ar & payoff;
	        ar & newPosition;
	        ar & oldPosition;
	        ar & team;
	        ar & otherTeam;
	        ar & occupant;
	        ar & occupantThatWasEaten;
	        ar & lastInputs;


	    }

	    PayONPos& operator=(const PayONPos& ps);
	    bool operator==(const PayONPos& ps);
	    bool operator!=(const PayONPos& ps);




};

//functor for max_element for PayONPos structs
struct PayONPosFunctor{
	bool operator()(PayONPos i, PayONPos j){return i.payoff<j.payoff;}
};


#endif

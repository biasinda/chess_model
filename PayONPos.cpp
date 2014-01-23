//testing a struct

#include "PayONPos.h"



PayONPos::PayONPos():newPosition(repast::Point<int> (0,0)), oldPosition(repast::Point<int> (0,0)){ }

PayONPos::PayONPos(double _newPayoff, repast::Point<int> _positionOld,  repast::Point<int> _positionNew, int _newTeam, int _newOccupant,int _newOccupantThatWasEaten, int _newOtherTeam, LastInputs _lastInputs):
payoff(_newPayoff), oldPosition(_positionOld), newPosition(_positionNew), team(_newTeam), occupant(_newOccupant), occupantThatWasEaten(_newOccupantThatWasEaten) ,otherTeam(_newOtherTeam), lastInputs(_lastInputs){ }

PayONPos& PayONPos::operator=(const PayONPos& ps){

	payoff=ps.payoff;
	newPosition=ps.newPosition;
	oldPosition=ps.oldPosition;
	team=ps.team;
	occupant=ps.occupant;
	otherTeam=ps.otherTeam;
	occupantThatWasEaten=ps.occupantThatWasEaten;
	lastInputs=ps.lastInputs;

	return *this;


}

bool PayONPos::operator==(const PayONPos& ps){

	if (payoff == ps.payoff && newPosition == ps.newPosition
			&& oldPosition == ps.oldPosition && team == ps.team
			&& occupant == ps.occupant && otherTeam == ps.otherTeam
			&& occupantThatWasEaten == ps.occupantThatWasEaten
			&& lastInputs == ps.lastInputs)
	{ return true;}
	else
	{return false;}

}

bool PayONPos::operator!=(const PayONPos& ps){

	if (payoff != ps.payoff || newPosition != ps.newPosition
			|| oldPosition != ps.oldPosition || team != ps.team
			|| occupant != ps.occupant || otherTeam != ps.otherTeam
			|| occupantThatWasEaten != ps.occupantThatWasEaten
			|| lastInputs != ps.lastInputs)
	{ return true;}
	else
	{return false;}

}

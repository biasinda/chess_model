#include "LastInputs.h"



LastInputs::LastInputs(){ }

LastInputs::LastInputs(int _sumTeam1Contacts, int _sumTeam2Contacts,  int _Occupant, int _willPandwillT0, int _willPandwillT1, int _qOccupant):
		sumTeam1Contacts(_sumTeam1Contacts), sumTeam2Contacts(_sumTeam2Contacts), Occupant(_Occupant), willPandwillT0(_willPandwillT0), willPandwillT1(_willPandwillT1), qOccupant(_qOccupant){ }



LastInputs& LastInputs::operator=(const LastInputs& li){
	sumTeam1Contacts=li.sumTeam1Contacts;
	sumTeam2Contacts=li.sumTeam2Contacts;
	Occupant=li.Occupant;
	willPandwillT0=li.willPandwillT0;
	willPandwillT1=li.willPandwillT1;
	qOccupant=li.qOccupant;

	return *this;


}

bool LastInputs::operator==(const LastInputs& li){
	if(sumTeam1Contacts == li.sumTeam1Contacts &&
	sumTeam2Contacts == li.sumTeam2Contacts &&
	Occupant == li.Occupant &&
	willPandwillT0 == li.willPandwillT0 &&
	willPandwillT1 == li.willPandwillT1 &&
	qOccupant == li.qOccupant)
	{return true;}
	else {return false;}

	}

bool LastInputs::operator!=(const LastInputs& li){
	if(sumTeam1Contacts != li.sumTeam1Contacts ||
	sumTeam2Contacts != li.sumTeam2Contacts ||
	Occupant != li.Occupant ||
	willPandwillT0 != li.willPandwillT0 ||
	willPandwillT1 != li.willPandwillT1 ||
	qOccupant != li.qOccupant)
	{return true;}
	else {return false;}

	}

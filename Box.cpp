//Box.cpp


/* this will be a box agent*/
// one of these will be present on each grid point and will have all information relevant to the payoff

//cpp file for Box agent

#include "Box.h"

//Box::Box(repast::AgentId id): id_, Occupant, Protectors, Threateners{ }

//constructors

Box::Box(repast::AgentId id): id_(id) { }

Box::Box(repast::AgentId id, int newOccupant): id_(id), Occupant(newOccupant){ }

Box::Box(repast::AgentId id, int newOccupant, int newTeam): id_(id), Occupant(newOccupant), team(newTeam){ }

Box::Box(repast::AgentId id, int newOccupant, std::vector<int> newTeam1Contacts,
		std::vector<int> newTeam2Contacts, int newTeam, std::vector<PayONPos> newBestMove, LastInputs newLastInputs) :
		id_(id), Occupant(newOccupant), team1Contacts(newTeam1Contacts), team2Contacts(
				newTeam2Contacts), team(newTeam), bestMove(newBestMove), lastInputs(newLastInputs) {
}

Box::~Box(){ }

//setters

void Box::set(int currentRank, int newOccupant, std::vector<int> newTeam1Contacts, std::vector<int> newTeam2Contacts, int newTeam, std::vector<PayONPos> newBestMove, LastInputs newLastInputs){
    id_.currentRank(currentRank);
    Occupant=newOccupant;
	team1Contacts=newTeam1Contacts;
	team2Contacts=newTeam2Contacts;
	team =newTeam;
	bestMove=newBestMove;
	lastInputs=newLastInputs;

}

void Box::setOccupant(int newOccupant){
	Occupant=newOccupant;
}

void Box::setTeam(int newTeam){
	team=newTeam;
}

void Box::setTeam1Contacts(std::vector< int> newTeam1Contacts){
	team1Contacts=newTeam1Contacts;
}

	void Box::addTeam1Contact(int newTeam1Contact){
		team1Contacts.push_back(newTeam1Contact);
	}

void Box::setTeam2Contacts(std::vector< int> newTeam2Contacts){
	team2Contacts=newTeam2Contacts;
}

	void Box::addTeam2Contact(int newTeam2Contact){
		team2Contacts.push_back(newTeam2Contact);
	}

	void Box::setBestMove(std::vector<PayONPos> newBestMove){
		bestMove=newBestMove;

	}


//getters

double Box::getPayoff(int qTeam, int qOccupant, double weights[5], std::vector<int> willPandwillT){


	double payoff;

	//if piece is himself a team1contact need to subtract him from team 1ctcs
	int nonprot1=0;
	int nonprot2=0;

	//issue is with pawn, pawn is only a contact if I am moving diagonally...

	if(qOccupant!=PAWNP){
		if(qTeam==1){
			nonprot1=qOccupant;
			nonprot2=0;
		}else if(qTeam==2){
			nonprot1=0;
			nonprot2=qOccupant;
		}

	}//end if(!=pawn)

	//if a pawn is trying to move, only remove him from t1cs if he is moving diagonally, ie the local occupant!=0

	if(qOccupant==PAWNP){

		if(Occupant!=0){

			if(qTeam==1){
				nonprot1=qOccupant;
				nonprot2=0;
			}else if(qTeam==2){
				nonprot1=0;
				nonprot2=qOccupant;
			}

		}////end occupant!=0

	}//end if(=pawn)




	//sum protectors and threateners
	int sumTeam1Contacts= std::accumulate(team1Contacts.begin(), team1Contacts.end(),0)-nonprot1;


	int sumTeam2Contacts=std::accumulate(team2Contacts.begin(), team2Contacts.end(),0)-nonprot2;

/*
		switch (qOccupant) { // get weights which depend on piece that is trying to move
		case PAWNP: //pawn

			weights[0] = 1;
			weights[1] = -1;
			weights[2] = 1;
			break;


		case KNIGHT: //knight

			weights[2] = 2;
			weights[3] = -2;
			weights[4] = 2;
			break;

		case BISHOP: //bishop

			weights[0] = 3;
			weights[1] = -3;
			weights[2] = 3;
			break;

		case ROOK: //rook

			weights[0] = 4;
			weights[1] = -4;
			weights[2] = 4;
			break;

		case QUEEN: //queen

			weights[0] = 5;
			weights[1] = -5;
			weights[2] = 5;
			break;

		case KING: //king

			weights[2] = 1;
			weights[3] = -10;
			weights[4] = 1;
			break;

}//end switch
*/

		//occupant weight is always postive sinc can only be of opposite team, ie cannot move to box occupied by same team...
		//wilPandWillT are already switched base on team, so do not need to switch again...
		//willPandWillT0 are pieces team1 will protect, and willPandwillT1 are pieces team 2 will protect

				//set the payoff
				payoff=weights[2]*sumTeam1Contacts+weights[3]*sumTeam2Contacts+weights[4]*Occupant+ weights[0]*willPandwillT[0] + weights[1]*willPandwillT[1];


		//want to store all the last payoff inputs in order to use for the reinforcement learning, these are then stores in the PayOnPos
		lastInputs.Occupant=Occupant;
		lastInputs.sumTeam1Contacts=sumTeam1Contacts;
		lastInputs.sumTeam2Contacts=sumTeam2Contacts;
		lastInputs.willPandwillT0=willPandwillT[0];
		lastInputs.willPandwillT1=willPandwillT[1];
		lastInputs.qOccupant=qOccupant;

		return payoff;

}//end getPayoff()



//removers
void Box::clearAllContacts(){
	(this->team2Contacts).clear();
	(this->team1Contacts).clear();


}

void Box::removeTeam2Contact(int oldTeam2Contact){
	//get the box team2Contacts
	std::vector<int> oldTeam2Contacts=this->team2Contacts;
	int contactToRemove=oldTeam2Contact;

	//find first occurrence of oldContact in Contacts
	std::vector<int>::iterator p=std::find(oldTeam2Contacts.begin(),oldTeam2Contacts.end(),contactToRemove);


	//remove the oldContact from the contacts
	oldTeam2Contacts.erase(p);

	this->team2Contacts=oldTeam2Contacts;

	std::cout<<"UPDATE CONTACTS: REMOVE1CONTACT erased \n";
}//end removeTeam2Contact

void Box::removeTeam1Contact(int oldTeam1Contact){
	//get the box team1Contacts
	std::vector<int> oldTeam1Contacts=this->team1Contacts;
	int contactToRemove=oldTeam1Contact;


	//find first occurrence of oldContact in Contacts
	std::vector<int>::iterator p=std::find(oldTeam1Contacts.begin(),oldTeam1Contacts.end(),contactToRemove);


	//remove the oldContact from the contacts
	oldTeam1Contacts.erase(p);

	this->team1Contacts=oldTeam1Contacts;

	std::cout<<"UPDATE CONTACTS: REMOVE1CONTACT erased \n";

}//end removeTeam1Contact

/* Serializable Box Package Data */

BoxPackage::BoxPackage(){ }

BoxPackage::BoxPackage(int _id, int _rank, int _type, int _currentRank, int _Occupant,  std::vector<int> _team1Contacts,  std::vector<int> _team2Contacts, int _team, std::vector<PayONPos> _bestMove, LastInputs _lastInputs):
id(_id), rank(_rank), type(_type), currentRank(_currentRank), Occupant(_Occupant), team1Contacts(_team1Contacts), team2Contacts(_team2Contacts), team(_team), bestMove(_bestMove), lastInputs(_lastInputs)  { }





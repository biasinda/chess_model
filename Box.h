//Box.h

//header file for box agent for chessModel

#ifndef BOX
#define BOX

#include "repast_hpc/AgentId.h"
#include "repast_hpc/SharedContext.h"
#include "repast_hpc/Point.h"


#include "numeric"
#include "algorithm"
#include "PayONPos.h"
#include "LastInputs.h"
#include "movePackage.h"


#define PAWNP 1
#define KNIGHT 2
#define BISHOP 3
#define ROOK 5
#define QUEEN 10
#define KING 20

/* Agents */
class Box{

private:
    repast::AgentId   id_;



public:

    //name and value of occupant of box
    int Occupant;

    //value of team (1 or 2 or 0(no occupant) of occupant
    int team;

    //names and values of protectors of box
    std::vector<int> team1Contacts;

    //names and values of threateners of box
    std::vector<int> team2Contacts;

    //type two agent has fields to store a best move:
    std::vector<PayONPos> bestMove;

    LastInputs lastInputs;



    Box(repast::AgentId id);
    Box(repast::AgentId id, int Occupant);

    Box(repast::AgentId id, int Occupant, int team);
    Box(repast::AgentId id, int Occupant, std::vector<int> Protectors, std::vector<int> Threateners, int team, std::vector<PayONPos> bestMove, LastInputs lastInputs );

    virtual ~Box();

    /* Required Getters */
    virtual repast::AgentId& getId(){                   return id_;    }
    virtual const repast::AgentId& getId() const {      return id_;    }

    /* Getters specific to this kind of Agent */
    	int getOccupant(){                                  			  return Occupant;      }
    	int getTeam(){                                  			  	  return team;          }

    	std::vector<int> getTeam1Contacts(){                                 return team1Contacts;    }

    	std::vector<int> getTeam2Contacts(){                                return team2Contacts;   }



    	//get method for payoff will return payoff depending on team and type of piece that is asking
     	double getPayoff(int qTeam, int qOccupant, double weights[5], std::vector<int> willPandwillT);



    /* Setters */
    void set(int currentRank, int newOccupant, std::vector<int> newTeam1Contacts, std::vector<int> newTeam2Contacts, int team, std::vector<PayONPos> bestMove, LastInputs lastInputs);


    void setTeam(int newTeam);

    void setOccupant(int newOccupant);

    void setBestMove(std::vector<PayONPos> bestMove);

	void setTeam1Contacts(std::vector<int> newTeam1Contacts);
	void addTeam1Contact(int newTeam1Contact);

	void setTeam2Contacts(std::vector<int> newTeam2Contacts);
	void addTeam2Contact(int newTeam2Contact);

	/* Removers*/
	void removeTeam1Contact(int oldTeam1Contact);

	void removeTeam2Contact(int oldTeam2Contact);

	void clearAllContacts();
	/*template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & boost::serialization::base_object<Box>(*this);
		ar & id_;
		ar & value;
		ar & total;
	}*/


};//end Box-agent




//to communicate agents between processes need to make the serializable...
/* Serializable Box Agent Package */
struct BoxPackage {

public:
    int    id;
    int    rank;
    int    type;
    int    currentRank;

    char cboard[8][8];

    //name and value of occupant of box
    int Occupant;
    int team;

    //names and values of protectors of box
    std::vector< int> team1Contacts;

    //names and values of threateners of box
    std::vector<  int> team2Contacts;

    std::vector<PayONPos> bestMove;
    LastInputs lastInputs;

    /* Constructors */
    BoxPackage(); // For serialization
    BoxPackage(int _id, int _rank, int _type, int _currentRank, int _Occupant,  std::vector<int> _newTeam1Contacts,  std::vector<int> _newTeam2Contacts, int _team, std::vector<PayONPos> _bestMove, LastInputs lastInputs);

    /* For archive packaging */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version){
        ar & id;
        ar & rank;
        ar & type;
        ar & currentRank;


        ar & Occupant;
        ar & team1Contacts;
        ar & team2Contacts;
        ar & team;

        ar & bestMove;
        ar & lastInputs;

    }

    repast::AgentId getId() const {
    	return repast::AgentId(id, rank, type);
       }

};//end serializable package Box

#endif

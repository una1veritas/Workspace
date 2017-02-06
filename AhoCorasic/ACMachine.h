/*
 * ACMachine.h
 *
 *  Created on: 2017/02/05
 *      Author: sin
 */

#ifndef ACMACHINE_H_
#define ACMACHINE_H_

#include <cinttypes>
#include <string>
#include <unordered_map>

typedef uint32_t uint32;
typedef int32_t int32;

typedef char Sigma;

class AhoCorasickMachine {
public:
	static const int32 AlpahetSize = sizeof(Sigma);
	static const int32 StatesLimit = 512;

private:
	typedef uint32 State;
	typedef std::pair<State,Sigma> StateSigmaPair;

	friend inline bool operator ==(StateSigmaPair const& lhs, StateSigmaPair const& rhs) {
		return (lhs.first == rhs.first) && (lhs.second == rhs.second);
	}
	struct StateSigmaPairHash {
	   size_t operator() (const StateSigmaPair &obj) const {
	     return (obj.first << 16) | (obj.second & 0xffff) | ((obj.first >> 16) & 0xffff) ;
	   }
	};

	std::unordered_map<StateSigmaPair,State,StateSigmaPairHash> trans;
	std::unordered_map<State,State> failure;
	std::unordered_map<State,std::vector<const std::string&> > out;
};


#endif /* ACMACHINE_H_ */

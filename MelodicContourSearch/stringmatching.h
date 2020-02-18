#ifndef _STRINGMATCHING_H_
#define _STRINGMATCHING_H_

#include <iostream>
#include <string>
#include <vector>

#define max(a, b) ((a) < (b) ? (b) : (a))
class naive {
	std::string pattern;
	unsigned int state;

public:
	naive(const char * str) : pattern(str), state(0) { }

	int size() const { return pattern.size(); }

	bool compare(const std::string & txt, const int & pos) {
		for(state = 0; state < pattern.size() && pattern[state] == txt[pos + state]; ++state) {}
		return state == pattern.size();
	}

	int find(const std::string & txt ) {
		unsigned int pos; // pos represents the place which will be matched to the last character of the pattern.
		for (pos = 0; pos < txt.size() - pattern.size() + 1; ++pos) {
			if ( compare(txt, pos) ) {
#ifdef FIND_DEBUG
				std::cout << "*" << std::endl;
#endif
				return pos;
			}
#ifdef FIND_DEBUG
			std::cout << "pos = " << pos << ", state = " << state << std::endl;
#endif
			}
		return pos;
	}

	// find all the occurrences.
	std::vector<int> find_all(const std::string & txt ) {
		std::vector<int> occurrences;
		unsigned int pos;
		for (pos = 0; pos < txt.size() - pattern.size() + 1; ++pos) {
			if ( compare(txt, pos) ) {
#ifdef FIND_DEBUG
				std::cout << "*";
#endif
				occurrences.push_back(pos);
			}
#ifdef FIND_DEBUG
			std::cout << "pos = " << pos << ", state = " << state << std::endl;
#endif
			}
		return occurrences;
	}

	friend std::ostream & operator<<(std::ostream & ost, const naive & pm) {
		ost << "naive('" << pm.pattern << "' ("<< pm.size() << ") )";
		return ost;
	}

};

// An implementation of Knuth-Morris-Pratt algorithm
class kmp {
	std::string pattern;
	std::vector<int> failure;
	unsigned int state;

private:
	void initialize() {
		unsigned int i, j;
		i = 1, j = 0, failure[0] = 0;
		while ( i < pattern.size() ) {
			if (pattern[i] == pattern[j]) {
				//std::cout << "+";
				j++;
				failure[i] = j;
				i++;
			} else {
				if (j > 0) {
					//std::cout << "-";
					j = failure[j - 1];
				} else {
					//std::cout << "0";
					failure[i] = 0;
					i++;
				}
			}
		}
	}

public:
	kmp(const char * str) : pattern(str), state(0) {
		failure.resize(pattern.size());
		initialize();
	}

	int find(const std::string & txt) {
		unsigned int pos = 0;
		state = 0;
		while (pos < txt.size() - pattern.size() + 1) {
#ifdef FIND_DEBUG
			std::cout << pos << ", " << state << std::endl;
#endif
			if ( txt[pos] == pattern[state]) {
				++state;
				++pos;
				if (state == pattern.size()) {
#ifdef FIND_DEBUG
					std::cout << "matched. " << pos << ", " << state << std::endl;
#endif
					state = failure[state-1];
					return pos - pattern.size();
				}
			} else {
				if ( state == 0 ) {
					++pos;
				} else {
					state = failure[state-1];
#ifdef FIND_DEBUG
					std::cout << "failed: " << pos << ", " << state << " (" << failure[state-1] << ") " << std::endl;
#endif
				}
			}
		}
		if ( pos < txt.size() - pattern.size() + 1 )
			return pos;
		return txt.size();
	}

	std::vector<int> find_all(const std::string & txt ) {
		std::vector<int> occurrs;
		unsigned int pos = 0;
		state = 0;
		while (pos < txt.size()) {
#ifdef FIND_DEBUG
			std::cout << pos << ", " << state << std::endl;
#endif
			if ( txt[pos] == pattern[state]) {
				state++;
				pos++;
				if (state == pattern.size()) {
					occurrs.push_back(pos - pattern.size());
					state = failure[state-1];
#ifdef FIND_DEBUG
					std::cout << "matched: " << pos << ", " << state << std::endl;
#endif
				}
			} else {
				if ( state == 0 ) {
					++pos;
				} else {
					state = failure[state-1];
#ifdef FIND_DEBUG
					std::cout << "failed: " << pos << ", " << state << " (" << failure[state-1] << ") " << std::endl;
#endif
				}
			}
		}
		return occurrs;
	}

	friend std::ostream & operator<<(std::ostream & ost, const kmp & p) {
		ost << "kmp('" << p.pattern << "' (" << p.pattern.size() << ") [";
		for(unsigned int i = 0; i < p.failure.size(); i++) {
			ost << p.failure[i];
			if ( i+1 < p.failure.size() )
				ost << ", ";
		}
		ost << "]) ";
		return ost;
	}
};

// An implementation of (Boyer-Moore-) Horspool algorithm
class horspool {
	std::string pattern;
	std::vector<unsigned int> delta;
	unsigned int state;

	void initialize() {
		delta.resize( 256 );
		for (int i = 0; i < 256; ++i)
			delta[i] = pattern.size();
		for (unsigned int i = 0; i < pattern.size() - 1; i++) {
			delta[pattern[i]] = pattern.size() - i - 1;
		}
	}

public:
	horspool(const char * pat) : pattern(pat), state(0) {
		initialize();
		return;
	}

	int size() const { return pattern.size(); }

	// find the 1st occurrence.
	// function for the outer-loop.
	int find(const std::string & txt) {
		unsigned int ix; // pos represents the place corresponds to the head of the pattern.
		for (ix = pattern.size() - 1; ix < txt.size(); ix += delta[txt[ix]]) {
			for (state = 0;
					state < pattern.size() && (pattern[pattern.size() - state - 1] == txt[ix - state]);
					++state);
			if ( state == pattern.size() ) {
#ifdef FIND_DEBUG
				std::cout << "*" << std::endl;
#endif
				return ix;
			}
#ifdef FIND_DEBUG
			std::cout << "pos = " << ix << ", state = " << state << ", delta["<< txt[ix] << "] = " << delta[txt[ix]] << std::endl;
#endif
		}
		return ix + 1 - pattern.size();
	}

	// find all the occurrences.
	std::vector<int> find_all(const std::string & txt) {
		std::vector<int> occurrences;
		unsigned int ix;
		for (ix = pattern.size() - 1; ix < txt.size(); ix += delta[txt[ix]]) {
			for (state = 0;
					state < pattern.size() && (pattern[pattern.size() - state - 1] == txt[ix - state]);
					++state);
			if ( state == pattern.size() ) {
#ifdef FIND_DEBUG
				std::cout << "*";
#endif
				occurrences.push_back(ix+1-pattern.size());
			}
#ifdef FIND_DEBUG
			std::cout << "pos = " << ix << ", state = " << state << ", delta["<< txt[ix] << "] = " << delta[txt[ix]] << std::endl;
#endif
		}
		return occurrences;
	}

	friend std::ostream & operator<<(std::ostream & ost, const horspool & hors) {
		int count = 0;
		ost << "horspool('" << hors.pattern << "' ("<< hors.size() << "), [";
		for (unsigned int i = 0; i < hors.delta.size(); i++) {
			if ( hors.delta[i] != hors.pattern.size() ) {
				if ( count != 0 )
					ost << ", ";
				ost << (char) i << ":" << hors.delta[i];
				++count;
			}
		}
		ost << "]) ";
		return ost;
	}

};

#endif /* _STRINGMATCHING_H_ */

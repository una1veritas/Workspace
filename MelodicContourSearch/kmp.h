/*
 * kmp.h
 *
 *  Created on: 2019/12/08
 *      Author: sin
 */

#ifndef KMP_H_
#define KMP_H_

typedef uint16_t uint16;
typedef uint32_t uint32;

class kmp {
	std::vector<char> pattern;
	std::vector<uint16> failure;
	unsigned int state;

private:
	void init_pattern(const char * str) {
		pattern.clear();
		const char * p = str;
		pattern.push_back('*');
		++p;
		for ( ; *p; ++p) {
			pattern.push_back(*p);
		}
	}

	void init_failure() {
		unsigned int i, j;
		i = 1, j = 0, failure[0] = 0;
		while ( i < pattern.size() ) {
			if (pattern[i] == pattern[j] or pattern[i] == '*' or pattern[j] == '*') {
				failure[i] = j + 1;
				j++;
				i++;
			} else {
				if (j > 0) {
					j = failure[j - 1];
				} else {
					i++;
				}
			}
		}
	}


public:
	kmp(const char * str) {
		init_pattern(str);
		failure.resize(pattern.size());
		init_failure();
		state = 0;
	}

	int search(const std::vector<int8> & txt) {
		unsigned int pos = 0;
		state = 0;
		while (pos < txt.size()) {
#ifdef KMP_DEBUG
			std::cout << pos << ", " << state << std::endl;
#endif //ifdef KMP_DEBUG
			if ( (pattern[state] == '*') ||
					(pattern[state] == '=' && txt[pos-1] == txt[pos]) ||
					(pattern[state] == '+' && txt[pos-1] < txt[pos]) ||
					(pattern[state] == '-' && txt[pos-1] > txt[pos]) ) {
				state++;
				pos++;
				if (state == pattern.size()) {
#ifdef KMP_DEBUG
					std::cout << "matched. " << pos << ", " << state << std::endl;
#endif //ifdef KMP_DEBUG
					return pos - pattern.size();
				}
			} else {
				if (state != 0) {
					state = failure[state-1];
#ifdef KMP_DEBUG
					std::cout << "failured " << state << std::endl;
#endif //ifdef KMP_DEBUG
				}
			}
		}
		return pos;
	}

	int search(const std::string & txt) {
		unsigned int pos = 0;
		state = 0;
		while (pos < txt.size()) {
#ifdef KMP_DEBUG
			std::cout << pos << ", " << state << std::endl;
#endif //ifdef KMP_DEBUG
			if ( txt[pos] == pattern[state]) {
				state++;
				pos++;
				if (state == pattern.size()) {
#ifdef KMP_DEBUG
					std::cout << "matched. " << pos << ", " << state << std::endl;
#endif //ifdef KMP_DEBUG
					return pos - pattern.size();
				}
			} else {
				if (state != 0) {
					state = failure[state-1];
#ifdef KMP_DEBUG
					std::cout << "failured " << state << std::endl;
#endif //ifdef KMP_DEBUG
				}
			}
		}
		return pos;
	}

	friend std::ostream & operator<<(std::ostream & ost, const kmp & p) {
		ost << "pmm('";
		for(unsigned int i = 0; i < p.pattern.size(); i++) {
			switch (p.pattern[i]) {
			case '=':
				ost << '=';
				break;
			case '+':
				ost << '+';
				break;
			case '-':
				ost << '-';
				break;
			case '*':
				ost << '*';
				break;
			}
		}
		ost << "' (" << p.pattern.size() << ") [";
		for(unsigned int i = 0; i < p.failure.size(); i++) {
			ost << p.failure[i];
			if ( i+1 < p.failure.size() )
				ost << ", ";
		}
		ost << "]) ";
		return ost;
	}

};


#endif /* KMP_H_ */

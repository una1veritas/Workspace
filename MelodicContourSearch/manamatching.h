/*
 * manamatching.h
 *
 *  Created on: 2020/02/13
 *      Author: sin
 */

#ifndef MANAMATCHING_H_
#define MANAMATCHING_H_

#include <string>
#include <vector>
#include <iostream>

struct manakmp {
	const std::string pattern;
	std::vector<unsigned int> f;

	void loopcheck() {
		const char *b = pattern.c_str();
		int i = 0;
		int j = 1;

		//f.resize(strlen(b) + 1);
		f[1] = 0;
		while (b[j] != '\0') {
			//std::cout << "b[" << i << "] = " << b[i] << ", b[" << j << "] = " << b[j] << "  ";
			if ( /*i == 0 ||*/ b[i] == b[j]) {
				f[++j] = ++i;
				//std::cout << "+";
			} else if ( i == 0 ){
				f[++j] = 0;
				//std::cout << "0";
			} else {
				i = f[i];
				//std::cout << "-";
			}
			/*
			std::cout << "[";
			for(int ix = 0; ix < f.size(); ++ix) {
				std::cout << f[ix] << " ";
			}
			std::cout << "]" << std::endl;
			*/
		}
		/*
		std::cout << "b[" << i << "] = " << b[i] << ", b[" << j << "] = " << (b[j] > ' ' ? b[j] : (int) b[j]) << "  ";
		std::cout << "[";
		for(int ix = 0; ix < f.size(); ++ix) {
			std::cout << f[ix] << " ";
		}
		std::cout << "]" << std::endl;
		*/
	}

public:
	manakmp(const std::string & p) : pattern(p) {
		f.resize(pattern.size() + 1);
		loopcheck();
	}

	unsigned int size() const { return pattern.size(); }

	unsigned  search(const std::string & text) const {
		return search(text.c_str(), text.size());
	}

	unsigned  search(const char * a) const {
		return search(a, strlen(a));
	}

	unsigned int search(const char *a, const unsigned int len) const {
		unsigned int i = 0;
		unsigned int j = 0;
		//const char * b = pattern.c_str();
		int frag = 0; // 1 まっちんぐちゅう。

		while ( j < pattern.size() ) {
			//std::cout << "i=" << i << ", j=" << j << std::endl;
			if ( !(i < len) ) {
				//std::cout << "reached to the end of t. " << i << std::endl;
				return i;
			}

			if (a[i] != pattern[j] && frag == 0) {
				i++;
			} else if (a[i] != pattern[j] && frag == 1) {
				if (f[j] > 0) {
					/*printf("j=%d, i=%d, f[i]=%d\n", j, i, f[i]);*/
					j = f[j];
				} else
					j = 0;
				frag = 0;
			} else {
				i++;
				j++;
				frag = 1;
			}
		}
		//printf("same \n");
		//  allfrag = 1;
		return i - pattern.size();
	}

	friend std::ostream & operator<<(std::ostream & ost, const manakmp & p) {
		ost << "manakmp('" << p.pattern << "' (" << p.pattern.size() << ") [";
		for(unsigned int i = 0; i < p.f.size(); i++) {
			ost << p.f[i];
			if ( i+1 < p.f.size() )
				ost << ", ";
		}
		ost << "]) ";
		return ost;
	}

};

#endif /* MANAMATCHING_H_ */

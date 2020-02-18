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
	std::vector<int> f;

	void loopcheck() {
		const char *b = pattern.c_str();
		int i = 0;
		int j = 1;

		//f.resize(strlen(b) + 1);
		f[1] = 0;
		while (b[j] != '\0') {
			std::cout << "b[" << i << "] = " << b[i] << ", b[" << j << "] = " << b[j] << "  ";
			if (i == 0 || b[i] == b[j])
				f[++j] = ++i;
			else
				i = f[i];
			std::cout << "[";
			for(int ix = 0; ix < f.size(); ++ix) {
				std::cout << f[ix] << " ";
			}
			std::cout << "]" << std::endl;
		}
		std::cout << "b[" << i << "] = " << b[i] << ", b[" << j << "] = " << (b[j] > ' ' ? b[j] : (int) b[j]) << "  ";
		std::cout << "[";
		for(int ix = 0; ix < f.size(); ++ix) {
			std::cout << f[ix] << " ";
		}
		std::cout << "]" << std::endl;
	}

public:
	manakmp(const std::string & p) : pattern(p) {
		f.resize(pattern.size() + 1);
		loopcheck();
	}

	unsigned int size() const { return pattern.size(); }

	bool search(const std::string & a) {
		unsigned int i = 0;
		unsigned int j = 0;
		int frag = 0; // 1 まっちんぐちゅう。

		//printf("pattern length = %d\n", strlen(b));
		//loopcheck(b);

		while (j < pattern.size()) {
			//printf("1 = %c",a[i]);
			//printf(" 2 = %c\n",b[j]);
			if (i < a.size() ) {
				//printf("not same \n");
				return false;
			}

			if (a[i] != pattern[j] && frag == 0) {
				i++;
			} else if (a[i] != pattern[j] && frag == 1) {
				if (f[i] > 1) {
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
		return true;
	}

	friend std::ostream & operator<<(std::ostream & ost, const manakmp & p) {
		ost << "manakmp('" << p.pattern << "' (" << p.pattern.size() << ") [";
		for(int i = 0; i < p.f.size(); i++) {
			ost << p.f[i];
			if ( i+1 < p.f.size() )
				ost << ", ";
		}
		ost << "]) ";
		return ost;
	}

};

#endif /* MANAMATCHING_H_ */

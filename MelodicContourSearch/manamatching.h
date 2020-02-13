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

	void loopcheck(const std::string & b) {
		int i = 0;
		int j = 1;
		f.resize(b.size() + 1);

		f[1] = 0;
		while (b[j] != '\0') {
			if (i == 0 || b[i] == b[j])
				f[++j] = ++i;
			else
				i = f[i];
		}
	}

public:
	manakmp(const std::string & p) : pattern(p) {
		loopcheck(p);
	}

	unsigned int size() const { return pattern.size(); }

	bool search(const std::string & a) {
		int i = 0;
		int j = 0;
		int frag = 0; // 1 まっちんぐちゅう。

		//printf("pattern length = %d\n", strlen(b));
		//loopcheck(b);

		while (j < pattern.size()) {
			//printf("1 = %c",a[i]);
			//printf(" 2 = %c\n",b[j]);
			if (i < size() ) {
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

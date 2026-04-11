/*
 *  stable --- prints the symbol table in alphabetical order
 */
#include <stdio.h>
#include "output.h"

void stable(const std::set<namedef> & list) {
	for (const auto & e : list) {
        printf("%-10s %04x\n", e.name, e.def);
    }
}

/*
 *  cross  --  prints the cross reference table 
 */

void cross(const std::set<namedef> &list) {
	int i = 1;
	link * tp = NULL;
	for (const auto &e : list) {
		printf("%-10s %04x *", e.name, e.def);
		tp = e.L_list;
		while (tp != NULL) {
			if (i++ > 10) {
				i = 1;
				printf("\n                      ");
			}
			printf("%04d ", tp->L_num);
			tp = tp->next;
		}
		printf("\n");
	}
}


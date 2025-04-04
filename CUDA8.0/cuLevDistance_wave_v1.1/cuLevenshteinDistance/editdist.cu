#include <stdio.h>
#include <stdlib.h>

#include "debug_table.h"


long dp_edist(char t[], long n, char p[], long m) {
	long * table;
	long result = 0;
	long del, repl, cellval;
	
	table = (long *) malloc(sizeof(long)*(m+1)*(n+1));
	if ( table == NULL )
		return n+m+1;
	
	// initialize cells in the top row or in the left-most column
	// n -- the number of columns, m -- the number of rows
	long col = 0;
	long row = 0;

	for(col = 0; col < n+1; ++col) {
		// row == 0
		table[(m + 1)*col] = col;
	}
	for(row = 1; row < m+1; ++row) {
		// col == 0
		table[row] = row;
	}

	//table calcuration
	for(long col = 1; col < n+1; col++) { // column, text axis
		for (long row = 1; row < m+1; row++) {  // row, pattern axis
			cellval = table[(m + 1)*(col - 1) + row] + 1;
			del = table[(m + 1)*col + row - 1] + 1;
			repl = table[(m + 1)*(col - 1) + row - 1] + (t[col-1] != p[row-1]);
			if (del < cellval )
				cellval = del;
			if (repl < cellval)
				cellval = repl;
			table[(m + 1)*col + row] = cellval;
		}
	}

#ifdef DEBUG_TABLE
	for (int i = 0; i < (n+1)*(m+1); i++) {
		debug_table[i] = table[i];
	}
#endif
	result = table[(n+1) * (m+1) - 1];
	free(table);
	return result;
}

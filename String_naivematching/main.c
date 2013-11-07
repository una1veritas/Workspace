#include <stdio.h>
#include <string.h>

/* Algorithm naive string matching */

int matching_naive(char * text, char * pattern, int start) {
	int pos, i;

<<<<<<< HEAD
	for(pos = start; text[pos] != 0; pos++) {
		for(i = 0; pattern[i] != 0; i++) {
			if ( text[pos+i] != pattern[i] )
				break;
		}
		if ( pattern[i] == 0 ) {
			printf("Found at %d,\n", pos);
			// return pos;
		}
	}
	printf("No more/any occurrences.\n");
=======
	for(pos = start; pos < n; pos++) {
		for(i = 0; i < m; i++) {
			// ... count the number of iterations here.
			if ( text[pos+i] != pattern[i] )
				break;
		}
		if ( !(i < m) ) {
			printf("Found at %d.\n", pos);
			return pos; // >= 0
		}
	}
	printf("No occurrences found.\n", pos);
	return -1;
>>>>>>> 9e15f2da1cd757996fe65bda6771546247f51abd
}

/* Algorithm end */

int main (int argc, const char * argv[]) {
	char * text = (char*) argv[1], * pattern = (char*)argv[2];
	printf("Text: %s\n", text);
	printf("Search pattern: %s\n\n", pattern);
	
	matching_naive(text, pattern, 0);

    return 0;
}

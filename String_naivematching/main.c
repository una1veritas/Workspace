#include <stdio.h>
#include <string.h>

/* Algorithm naive string matching */

int matching_naive(char * text, char * pattern, int start) {
	int n = strlen(text);
	int m = strlen(pattern);
	int pos, i;

	for(pos = start; pos < n; pos++) {
		for(i = 0; i < m; i++) {
			if ( text[pos+i] != pattern[i] )
				break;
		}
		if ( !(i < m) ) {
			printf("Found at %d,\n", pos);
			// return pos;
		}
	}
}

/* Algorithm end */

int main (int argc, const char * argv[]) {
	char * text = (char*) argv[1], * pattern = (char*)argv[2];
	printf("Text: %s\n", text);
	printf("Search pattern: %s\n\n", pattern);
	
	matching_naive(text, pattern, 0);

    return 0;
}

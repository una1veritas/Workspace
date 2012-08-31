#include <stdio.h>

int main (int argc, const char * argv[]) {
	int pos, i;
	char * text = (char*) argv[1], * pattern = (char*)argv[2];
	printf("Text: %s\n\n", text);
	printf("Pattern to search: %s\n\n", pattern);
	
	pos = 0;
from_pattern_beginning:
	i = 0;
to_next_char:
	if ( text[pos] == 0 ) 
		goto end_procedure;
	if ( pattern[i] == 0 ) {
		printf("Found at %d,\n", pos);
		pos++;
		goto from_pattern_beginning;
	} 
	if ( pattern[i] != text[pos+i] ) {
		pos++;
		goto from_pattern_beginning;
	} else {
		i++;
	}
	goto to_next_char;	
end_procedure:
    return 0;
}

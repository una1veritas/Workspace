#include <iostream>

/* Algorithm Insertion Sort */

typedef char Data;
#define Order(x,y)  (x < y)

void sort_insertion(Data array[], int num, int sorted = 0) {
	int i, j;
	Data key;

	if ( sorted == 0 )
		sorted = 1; // any sequence of length one is a sorted sequence.
	for ( j = sorted; j < num; j++) {
		key = array[j];
		i = j - 1;
		while ( (i >= 0) && Order(key, array[i]) ) {
			array[i+1] = array[i];
			i = i - 1;
		}
		array[i+1] = key;
	}
}

/* Algorithm end */

int main (int argc, char * const argv[]) { 
	
	Data * a, * p;
	int length;

	// accepting input
	if ( argv[1] == NULL ) {
		std::cout << "No input chars." << std::endl;
		return 0;
	} else {
		a = argv[1];
		length = 0;
		for (p = a; *p != 0 ; p++, length++ );
		std::cout << "Input: " << a << ", " << std::endl
				<< "Length: " << length << std::endl;
	}
	
	sort_insertion(a, length, 0);

	std::cout << "Result \"" << a << "\"" << std::endl;
	
    return 0;
}

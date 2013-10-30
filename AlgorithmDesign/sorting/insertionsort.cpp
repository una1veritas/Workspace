#include <iostream>

typedef unsigned long Data;

void sort_insertion(Data a[], int num, int sorted = 0) {	// sort the input sequence
	int i, j;
	Data key;

	for ( j = 1; j < length; j++) {
		key = A[j];
		i = j - 1;
		while ( (i >= 0) && A[i] > key) {
			A[i+1] = A[i];
			i = i - 1;
		}
		A[i+1] = key;
	}
}

int main (int argc, char * const argv[]) { 
	
	char * A, * p;
	int length;

	// accepting input
	if ( argv[1] == NULL ) {
		std::cout << "No input chars." << std::endl;
		//return 0;
	} else {
		A = argv[1];
		length = 0;
		for (p = A; *p != 0 ; p++, length++ );
		std::cout << "Input: " << A << ", " << std::endl
				<< "Length: " << length << std::endl;
	}
	

	std::cout << "Result \"" << A << "\"" << std::endl;
	
    return 0;
}

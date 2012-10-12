#include <iostream>


int main (int argc, char * const argv[]) { 
	// insert code here...
	
	char * inputs, * p;
	int length;
	if ( argv[1] == NULL ) {
		cout << "No input chars." << std::endl;
		//return 0;
	} else {
		inputs = argv[1];
		length = 0;
		for (p = inputs; *p != 0 ; p++, length++ );
		cout << "Input: " << inputs << "Length: " << length << std::endl;
	}
	
	//
	
	int i, top;
	int sortedEnd;
	char c;
	for ( sortedEnd = 0; sortedEnd < length; sortedEnd++) {
		top = sortedEnd;
		for (i = sortedEnd; i < length; i++ ) {
			if ( inputs[i] < inputs[top] ) {
				top = i;
			}
		}
		c = inputs[sortedEnd];
		inputs[sortedEnd] = inputs[top];
		inputs[top] = c;
	}
	
	println();
	println(inputs);
	
    return 0;
}

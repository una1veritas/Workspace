#include <iostream>
using namespace std;

int main (int argc, char * const argv[]) { 
	char * array, * p;
	int length;

	if ( argv[1] == NULL ) {
		cout << "No input." << std::endl;
		return 1;
	}

	array = argv[1];
	length = 0;
	for (p = array; *p != 0 ; p++, length++ );
	cout << "Input: " << array << endl
			<< "Length: " << length << endl;
	
	// sort routine
	
	int i, top;
	int sortedEnd;
	char c;
	for ( sortedEnd = 0; sortedEnd < length; sortedEnd++) {
		top = sortedEnd;
		for (i = sortedEnd; i < length; i++ ) {
			if ( array[i] < array[top] ) {
				top = i;
				cout << "<";
			} else {
				cout << ">";
			}
		}
		c = array[sortedEnd];
		array[sortedEnd] = array[top];
		array[top] = c;
	}
	cout << endl;
	
	cout << endl ;
	cout << "Result: " << array << endl;
	
    return 0;
}

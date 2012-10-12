#include <iostream>

using namespace std;

int main(int argc, char * argv[]) {
	long a, b;
	long c;
	cout << "input two numbers." << endl;
	cin >> a >> b;

step1:
	c = a % b;
step2:
	if ( c == 0 ) {
		cout << "gcd = " << b << endl;
		goto step4;
	}
step3:
	a = b; b = c;
	goto step1;

step4:
  return (int) b;
}

#include <iostream>
#include <iomanip>

using namespace std;

int main(int argc, char * argv[]) {
  int y, m, d;
  int a, b;

  double jd;

  cout << "Hello." << endl;

  cin >> y; // = atoi(argv[1]);
  cin >> m; // = atoi(argv[2]);
  cin >> d; // = atoi(argv[3]);

  cout << y << "/" << m << "/" << d << endl;

  if ( m <= 2 ) {
    y -= 1;
    m += 12;
  }
  a = int(y/100);  b = 2 - a + (a/4);
  jd = int(365.25 * y) + int(30.6001 * (m+1)) + d + 1720994.5 + b;

  cout << "jd = " << setprecision (1) << fixed << jd << endl;
  cout << "Day of Week = " << int(jd + 1.5) % 7 << endl;

  return 0;
}

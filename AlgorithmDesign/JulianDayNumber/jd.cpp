#include <iostream>
#include <iomanip>

using namespace std;

int main(int argc, char * argv[]) {
  int y, m, d;
  int a, b;

  double jd;
  int wday;

  cout << "Hello." << endl;

  cin >> y;
  cin >> m;
  cin >> d;

  cout << y << "/" << m << "/" << d << endl;

  if ( m <= 2 ) {
    y -= 1;
    m += 12;
  }
  a = int(y/100);  b = 2 - a + (a/4);
  jd = int(365.25 * y) + int(30.6001 * (m+1)) + d + 1720994.5 + b;

  wday = int(jd + 1.5) % 7;

  cout << "jd = " << setprecision (1) << fixed << jd << endl;
  cout << "Day of Week = " << wday << endl;

  return 0;
}

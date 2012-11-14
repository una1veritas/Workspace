#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char * argv[]) {
  vector<int> val;
  int budget;

  unsigned int i, b, price;

  cin >> budget;
  if ( !(budget > 0) ) {
    cerr << "No money?" << endl;
    exit(1);
  }
  cout << "Your budget is " << budget << ". " << endl;
  
  for ( cin >> price; ! cin.eof() ; cin >> price) {
    if ( price == 0 ) break;
    val.push_back(price);
  }

  cout << "Prices of items: " << endl;
  for (vector<int>::iterator i = val.begin(); i != val.end(); i++) {
    cout << *i << ", ";
  }
  cout << endl << endl;

  /*
   * DP calc.
   */
  int best[val.size()][budget+1];
  
  // Base phase.
  for (b = 0; b <= budget; b++) {
    if ( b >= val[0] ) 
      best[0][b] = val[0];
    else
      best[0][b] = 0;
    //
    cout << best[0][b] << ", ";
  }
  cout << endl;

  // Recurrent phase.
  for (i = 0; i+1 < val.size(); i++) {
    for (b = 0; b <= budget; b++) {
      if ( b - val[i+1] >= 0 ) {
	best[i+1][b] = ( best[i][b] > best[i][b-val[i+1]] + val[i+1] ) ?
	  best[i][b] : (best[i][b-val[i+1]] + val[i+1]);
      } else {
	best[i+1][b] = best[i][b];
      }
      cout << best[i+1][b] << ", ";
    }
    cout << endl;
  }
  cout << endl;

  vector<int> buylist;

  for (i = val.size() - 1, b = budget; i > 0 ; i-- ) {
    cout << "(" << i << ", " << b << ") ";
    if ( best[i][b] == best[i - 1][b] ) {
      //cout << "(skip " << i << "th) ";
    } else {
      //cout << i << " (" << val[i] << " yen), ";
      buylist.push_back(i);
      b = b - val[i];
    }
  }
  cout << "(" << 0 << ", " << b << ") ";
  if (best[0][b] != 0)
    buylist.push_back(0);

  cout << endl << "Buy-list: ";
  for (vector<int>::iterator i = buylist.end(); i > buylist.begin(); ) {
    i--;
    cout << *i << ", ";
  }
  cout << "totally " << best[val.size()-1][budget] << " yen." << endl;
  
  return 0;
}

#include <iostream>

using namespace std;


void nneighbor(int size, float ** dist, int * tour) {
  int i, j, min, tmp;

  for (i = 0; i < size; i++)
    tour[i] = i;

  for (i = 0; i + 1 < size; i++) {
    min = i + 1;
    for (j = min + 1; j < size; j++) {
      //cout << "j = " << j << endl;
      if ( dist[tour[i]][tour[min]] > dist[tour[i]][tour[j]] )
	min = j;
    }
    //cout << "min = " << min << endl;
    tmp = tour[i+1];
    tour[i+1] = tour[min];
    tour[min] = tmp;

    //for (tmp = 0; tmp < size; tmp++) 
    //  cout << tour[tmp] << ", ";
    //  cout << endl;
  }

}

int main(void ) {

  float tmp;
  int sz, i, j;

  cin >> sz;

  float ** D;

  D = (float **) malloc(sizeof(float *) * sz);
  for (i = 0; i < sz; i++)
    D[i] = (float *) malloc(sizeof(float[sz]));
  int tour[sz];

  for (i = 0; i < sz && (! cin.eof()); i++) {
    for (j = 0; j < sz && (! cin.eof()); j++) {
      cin >> D[i][j];
      //cout << D[i][j] << ", ";
    }
    //cout << endl;
  }


  // *** //


  nneighbor(sz, D, tour);

  // *** //

  cout << "Tour: ";
  for (i = 0; i < sz; i++)
    cout << tour[i] << ", ";
  cout << endl;

  cout << "Total tour length: ";
  for (i = 0, tmp = 0; i < sz; i++)
    tmp += D[i][(i + 1) % sz];
  cout << tmp << "." << endl;

  return 0;
}

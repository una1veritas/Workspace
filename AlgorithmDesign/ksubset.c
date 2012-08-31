#include <stdio.h>

int main() {
  int n = 12;
  int i,k;
  int s[n+1];

  int t;

  for (k = 1; k <= n; k++) {
    for (t = 1; t <= k; t++)
      s[t] = t;
    i = k;

    //
    while ( i == k ) {
      printf("\n");
      for (t = 1; t <= k; t++) printf("%d, ", s[t]);
      printf("\n");
      
      while (1) {
	if ( s[i] < n - k + i ) {
	  s[i]++;
	  if ( i < k ) {
	    s[i+1] = s[i];
	    i++;
	  } else 
	    break;
	} else {
	  i--;
	  if ( i == 0 )
	    break;
	}
      }
    }
  }

}

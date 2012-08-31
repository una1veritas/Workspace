#include <stdio.h>
#include <stdlib.h>

/* prototypes */
void make_heap(int a[], int n);
void down_to_leaf(int a[], int i, int n);
void heap_sort(int a[], int n);


void make_heap(int a[], int n) {
  int i;
  for ( i = n / 2 - 1; ! (i < 0) ; i--) {
    down_to_leaf(a, i, n);
  }
  return;
}

void down_to_leaf(int a[], int i, int n) {
  int j;
  int t;
  
  while ( 2*i + 1 < n) {       // the i-th node has a child
	// counter++; 
    j = 2*i+1;                 // try the left child as default 
    if ( 2*i+2 < n )           // the i-th node has the right child 
      if ( a[2*i+1] < a[2*i+2] ) 
         j = 2*i+2;
    if (! (a[j] <= a[i]) ) { 
      t = a[j];                // swap a[i] and a[c]
      a[j] = a[i];
      a[i] = t;
      i = j;
    } else {
      break;
    }
  }
  return;
}

void heap_sort(int a[], int n) {
  int i;
  int t;

  make_heap(a, n);
  for (i = n - 1; i > 0; i--) {
    t = a[i];
    a[i] = a[0];
    a[0] = t;
    down_to_leaf(a, 0, i);
  }
  return;
}

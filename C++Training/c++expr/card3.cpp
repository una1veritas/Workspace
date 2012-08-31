#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum {
  spade = 0,
  diamond, 
  heart, 
  club
};

struct card {
private:
  int number;
  int suit;

public:

  card() {
    number = 1;
    suit = spade;
  }

  int suitVal(int s) {
    return suit = s;
  }
  
  int numberVal(int n) {
    return number = n;
  }

  void print() {
    printf("[");
    switch (number) {
    case 1:
      printf("A ");
      break;
    case 11:
      printf("J ");
      break;
    case 12:
      printf("Q ");
      break;
    case 13:
      printf("K ");
      break;
    default:
      printf("%d ", number);
      break;
    }
    printf("of ");
    switch (suit) {
    case spade:
      printf("Spade");
      break;
    case club:
      printf("Club");
      break;
    case heart:
      printf("Heart");
      break;
    case diamond:
      printf("Diamond");
      break;
    }
    printf("] ");
  }
  
};

struct deck {
  struct card pile[52];

  void shuffle() {
    int t, i, j;
    struct card swap;
    time_t seed;
    
    time(&seed);
    srandom(seed);
    for (t = 0; t < 100; t++) {
      i = random() % 52;
      j = ((random() % 51) + 1 + i) % 52;
      swap = pile[i];
      pile[i] = pile[j];
      pile[j] = swap;
    }
  }
  void print() {
    int i;
    for (i = 0; i < 52; i++) {
      pile[i].print();
      printf(", ");
    }
    printf("\n");
  }

};


int main(int argc, char * argv[]) {
  struct deck myDeck;
  int n, i, s;
  struct card * c;
  
  c = new card();
  c->print();

  printf("\n");

  i = 0;
  for (s = spade; s <= club; s++) {
    for (n = 1; n <= 13; n++) {
      myDeck.pile[i].suitVal(s);
      myDeck.pile[i].numberVal(n);
      i++;
    }
  }

  myDeck.shuffle();
  myDeck.print();

  return 0;
}

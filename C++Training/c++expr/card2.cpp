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
  int number;
  int suit;

  void print() {
    printf("[");
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
    printf(" ");
    switch (number) {
    case 1:
      printf("A");
      break;
    case 11:
      printf("J");
      break;
    case 12:
      printf("Q");
      break;
    case 13:
      printf("K");
      break;
    default:
      printf("%d", number);
      break;
    }
    printf("]");
  }
};

struct deck {
  struct card cards[52];

  void shuffle() {
    int t, i, j;
    struct card swap;
    time_t seed;
    
    time(&seed);
    srandom(seed);
    for (t = 0; t < 100; t++) {
      i = random() % 52;
      j = ((random() % 51) + 1 + i) % 52;
      swap = cards[i];
      cards[i] = cards[j];
      cards[j] = swap;
    }
  }

  void print() {
    int i;
    for (i = 0; i < 52; i++) {
      cards[i].print();
      printf(", ");
    }
    printf("\n");
  }
};


int main(int argc, char * argv[]) {
  struct deck myDeck;
  int n, i, s;

  i = 0;
  for (s = spade; s <= club; s++) {
    for (n = 1; n <= 13; n++) {
      myDeck.cards[i].suit = s;
      myDeck.cards[i].number = n;
      i++;
    }
  }

  myDeck.shuffle();

  for (i = 0; i < 52; i++) {
    myDeck.cards[i].print();
    printf(", ");
  }
  printf("\n");
  return 0;
}

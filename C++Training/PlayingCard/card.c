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
};

struct deck {
  struct card cards[52];
};

void shuffle_deck(struct deck * d) {
  int t, i, j;
  struct card swap;
  time_t seed;

  time(&seed);
  srandom(seed);
  for (t = 0; t < 200; t++) {
    i = random() % 52;
    j = ((random() % 51) + 1 + i) % 52;
    swap = d->cards[i];
    d->cards[i] = d->cards[j];
    d->cards[j] = swap;
  }
}

void print_card(struct card * c) {
  printf("[");
  switch (c->suit) {
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
  switch (c->number) {
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
      printf("%d", c->number);
      break;
    }
    printf("]");
}

void print_deck(struct deck * d) {
  int i;
  for (i = 0; i < 52; i++) {
    print_card( & d->cards[i] );
    printf(", ");
  }
  printf("\n");
}

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

  shuffle_deck( & myDeck );
  print_deck( & myDeck);

  return 0;
}
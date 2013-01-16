#include <stdio.h>
#include <stdlib.h>

/* this number includes the dummy header cells */
#define keyval_min 0
#define keyval_max 500

#define min(a,b) ((a)>(b)? (b) : (a))

struct Data {
	long value;
};
typedef struct Data Data;

long key(Data * d) {
	return d->value;
}

typedef struct Capsul Capsul;
struct Capsul {
	Data * dataptr;
	Capsul * next;
};

struct Buckets {
    Capsul * bucket[keyval_max - keyval_min];
    int tally;
};
typedef struct Buckets  Buckets;


void Buckets_init(Buckets * b) {
    int i;
    for (i = keyval_min; i < keyval_max; i++) {
        b->bucket[i - keyval_min] = NULL;
    }
    b->tally = 0;
}

void Buckets_add(Buckets * b, Data * d) {
	if ( key(d) < keyval_min || key(d) > keyval_max ) {
		printf("error: key %ld is out of range. ", key(d));
		return;
	}
	Capsul * newcapsul = (Capsul*) malloc(sizeof(Capsul));
	newcapsul->dataptr = d;
	newcapsul->next = b->bucket[key(d)];
    b->bucket[key(d)] = newcapsul;
    b->tally++;
}

void Buckets_dispose(Buckets * b) {
	// must be free all allocated memory areas...
}


int main (int argc, const char * argv[]) {
    int i;
    int aValue;
    Data * dataptr;
    Buckets buckets;
    // the first table_size buckets as the links to the list
    Buckets_init(&buckets);
    // adding all the elements to buckets
    printf("Adding data with... ");
    for (i = 1; i < argc; i++) {
        aValue = atol( argv[i]);
        printf("%ld, ", aValue);

        dataptr = (Data *) malloc(sizeof(Data));
        dataptr->value = aValue;
        Buckets_add(&buckets, dataptr);
    }
    printf("\nfinished.\n\n");

    // show contents of all the buckets
    Capsul * current;
    for (i = 0; i < keyval_max - keyval_min; i++) {
        for ( current = buckets.bucket[i]; current != NULL; current = current->next) {
        	printf("%ld, ", current->dataptr->value);
        };
        if ( buckets.bucket[i] != NULL) printf("\n");
    }
    Buckets_dispose(&buckets);

    return 0;
}

long hashcode(char * s) {
   // return s[0];

    long tmp = 0;
    int i;
    for (i = 0; s[i] != '\0' ; i++)
        tmp = s[i] + (tmp<<7);
    return 5;

}

#include <stdio.h>

/* this number includes the dummy header cells */
#define max_number 500
#define table_size 127
#define min(a,b) ((a)>(b)? (b) : (a))

typedef struct {
    void * element[max_number];
    int next[max_number];
    int tally;
} Buckets;

void Buckets_init(Buckets * b, int number) {
    int i;
    for (i = 0; i < min(number,max_number); i++) {
        b->next[i] = 0;
        b->element[i] = (void*) 0;
    }
    b->tally = number + 1;
}

void Buckets_add(Buckets * b, int * head, void * elem) {
    b->element[b->tally] = elem;
    b->next[b->tally] = *head;
    *head = b->tally;
    b->tally++;
}

int Buckets_next(Buckets * b, int current) {
    return b->next[current];
}

void * Buckets_element(Buckets * b, int index) {
    return b->element[index];
}

long hashcode(char * s);

int main (int argc, const char * argv[]) {
    int i, current;
    char * elem;
    Buckets buckets;
    // the first table_size buckets as the links to the list
    Buckets_init(&buckets, table_size);
    // adding all the elements to buckets
    for (i = 0; i < argc; i++) {
        elem = (void*) argv[i];
        //Buckets_add(&buckets, &( buckets.next[*elem] ), elem );
        Buckets_add(&buckets,&(buckets.next[(hashcode(elem) % (table_size-1)) + 1]),elem);
    }
    // show contents of all the buckets
    for (i = 0; i < table_size; i++) {
        current = i;
        if ( current = Buckets_next(&buckets,current) )
            printf("\nBucket %d", current);
        for ( ; ;current = Buckets_next(&buckets,current) ) {
            if ( Buckets_element(&buckets,current) != 0 )
                printf(", \"%s\" (%d) ",(char*) Buckets_element(&buckets,current), current);
            if ( Buckets_next(&buckets, current) == 0  )
                break;
        };
    }
    
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

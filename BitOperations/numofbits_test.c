/*
 *  File: numofbits_test.c
 *
 *  Copyright (C) NAKAMURA Minoru <nminoru@nminoru.jp>
 *
 *  $ gcc -g -Wall -mtune=core2 -O3 numofbits_test.c numofbits_routines.c
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>


enum {
    REPEAT    = 1000000000,
    DATA_SIZE = 65536
};


extern int dummy(uint32_t x) __attribute__((noinline));
extern int numofbits1(uint32_t x) __attribute__((noinline));
extern int numofbits2(uint32_t x) __attribute__((noinline));
extern int numofbits3(uint32_t x) __attribute__((noinline));
extern int numofbits4(uint32_t x) __attribute__((noinline));
extern int numofbits5(uint32_t x) __attribute__((noinline));
extern int popcnt(uint32_t x) __attribute__((noinline));


static double getdiff(const struct timeval *before, const struct timeval *after)
{
    return (after->tv_sec - before->tv_sec) + 1.0 * (after->tv_usec - before->tv_usec) / 1000000;
}

static uint32_t data[DATA_SIZE];

int main(int argc, char** argv)
{
    int i;

    // Prepare input datas
    for (i=0 ; i<REPEAT ; i++) {
        data[i % DATA_SIZE] = rand();
    }

    // Warm up data cache
    for (i=0 ; i<REPEAT ; i++) {
        dummy(data[i % DATA_SIZE]);
    }

    struct timeval before, after;

    // dummy
    gettimeofday(&before, NULL);
    for (i=0 ; i<REPEAT ; i++) {
        dummy(data[i % DATA_SIZE]);
    }
    gettimeofday(&after, NULL);
    double base_time = getdiff(&before, &after);

    // numofbits1
    gettimeofday(&before, NULL);
    for (i=0 ; i<REPEAT ; i++) {
        numofbits1(data[i % DATA_SIZE]);
    }
    gettimeofday(&after, NULL);
    printf("numofbits1: %10.6f sec\n", getdiff(&before, &after) - base_time);

    // numofbits2
    gettimeofday(&before, NULL);
    for (i=0 ; i<REPEAT ; i++) {
        numofbits2(data[i % DATA_SIZE]);
    }
    gettimeofday(&after, NULL);
    printf("numofbits2: %10.6f sec\n", getdiff(&before, &after) - base_time);

    // numofbits3
    gettimeofday(&before, NULL);
    for (i=0 ; i<REPEAT ; i++) {
        numofbits3(data[i % DATA_SIZE]);
    }
    gettimeofday(&after, NULL);
    printf("numofbits3: %10.6f sec\n", getdiff(&before, &after) - base_time);

    // numofbits4
    gettimeofday(&before, NULL);
    for (i=0 ; i<REPEAT ; i++) {
        numofbits4(data[i % DATA_SIZE]);
    }
    gettimeofday(&after, NULL);
    printf("numofbits4: %10.6f sec\n", getdiff(&before, &after) - base_time);

    // numofbits5
    gettimeofday(&before, NULL);
    for (i=0 ; i<REPEAT ; i++) {
        numofbits5(data[i % DATA_SIZE]);
    }
    gettimeofday(&after, NULL);
    printf("numofbits5: %10.6f sec\n", getdiff(&before, &after) - base_time);

    // popcnt (SSE 4.2)
    gettimeofday(&before, NULL);
    for (i=0 ; i<REPEAT ; i++) {
        popcnt(data[i % DATA_SIZE]);
    }
    gettimeofday(&after, NULL);
    printf("popcnt: %10.6f sec\n", getdiff(&before, &after) - base_time);

    return 0;
} 

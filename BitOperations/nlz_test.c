/*
 *  File: nlz_test.c
 *
 *  Copyright (C) NAKAMURA Minoru <nminoru@nminoru.jp>
 *
 *  $ gcc -g -Wall -mtune=core2 -O3 nlz_test.c nlz_routines.c
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

#include "nlz_routines.h"

enum {
    REPEAT    = 1000000000,
    DATA_SIZE = 65536
};


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

    // nlz1
    gettimeofday(&before, NULL);
    for (i=0 ; i<REPEAT ; i++) {
        nlz1(data[i % DATA_SIZE]);
    }
    gettimeofday(&after, NULL);
    printf("nlz1: %10.6f sec\n", getdiff(&before, &after) - base_time);

    // nlz2
    gettimeofday(&before, NULL);
    for (i=0 ; i<REPEAT ; i++) {
        nlz2(data[i % DATA_SIZE]);
    }
    gettimeofday(&after, NULL);
    printf("nlz2: %10.6f sec\n", getdiff(&before, &after) - base_time);

    // nlz3
    gettimeofday(&before, NULL);
    for (i=0 ; i<REPEAT ; i++) {
        nlz3(data[i % DATA_SIZE]);
    }
    gettimeofday(&after, NULL);
    printf("nlz3: %10.6f sec\n", getdiff(&before, &after) - base_time);

    // nlz4
    gettimeofday(&before, NULL);
    for (i=0 ; i<REPEAT ; i++) {
        nlz4(data[i % DATA_SIZE]);
    }
    gettimeofday(&after, NULL);
    printf("nlz4: %10.6f sec\n", getdiff(&before, &after) - base_time);

    // nlz5
    gettimeofday(&before, NULL);
    for (i=0 ; i<REPEAT ; i++) {
        nlz5(data[i % DATA_SIZE]);
    }
    gettimeofday(&after, NULL);
    printf("nlz5: %10.6f sec\n", getdiff(&before, &after) - base_time);

    return 0;
} 

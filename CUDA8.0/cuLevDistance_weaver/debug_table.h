#pragma once

#define DEBUG_TABLE

#ifdef DEBUG_TABLE
extern long * debug_table;
#endif

void show_table(long * table, long n, long m);
long compare_table(long * t0, long * t1, long n, long m);
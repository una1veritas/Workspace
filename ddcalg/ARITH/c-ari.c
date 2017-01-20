/*
 *	c-ari.c 共通部分
 *			(c) Nobukazu Yoshioka Oct. 23 1993
 */

#include <stdio.h>
#include <stdlib.h>

#include "c-ari.h"
#include "bitio.h"

extern long X, Y;		/* 区間 */
extern long Code[];

long u = 10000L;		/* u+CODE_SIZE <= 2^w で無ければならない */
int w = 15;

/******************************************************************
 *	累積頻度をCodeに設定する
 */
int make_Fi(void)
{
    int i;
    long total = 0;		/* 頻度の合計 */
    long old, frq, sum;
    int add = 0;		/* uへの追加分 */

    for (i = 0; i < CODE_SIZE; i++)
	total += Code[i];	/* 合計をとる */

    old = sum = add = 0;
    for (i = 0; i < CODE_SIZE; i++) {
	frq = Code[i];
	sum += frq;
	Code[i] = (double) sum *u / (double) total + add;
	if (frq != 0 && Code[i] == old) {	/* 頻度が少なすぎる */
	    Code[i] += 1;
	    add++;
	}
	old = Code[i];
    }

    u = Code[CODE_SIZE - 1];
    if (u > (0x01L << w)) {	/* uの値が大きすぎる */
	fprintf(stderr, "2^w(%ld) is smaller than u(%ld)\n",
		(0x01L << w), u);
	return -1;
    }
    return 0;
}

/****************************************************************
 *	ビット演算
 */

/*
 *	下位からmビット目を返す
 */
int bit(long x, int m)
{
    return ((x & (0x01L << (m - 1))) == 0) ? 0 : 1;
}

/*
 *	xのmビット以下のみを取り出す
 */

long bit_omit(long x, int m)
{
    long mask = 0x01L;

    while (--m > 0)
	mask |= (mask << 1);

    return (x & mask);
}

/*******************************************************
 *	ビット入出力
 */

/*
 *	n個のbitを出力する
 */
int bits_out(FILE * fp, int n, int bit)
{
    while (n-- > 0)
	bit_out(fp, bit);

    return n;
}

/*
 *		EOFで1を返し続ける
 */
int bit_input(FILE * fp)
{
    int c;

    c = bit_in(fp);
    return (c == -1) ? 1 : c;
}

/*
 *		算術圧縮 復号化ルーチン(Jones符号)
 *			(c) Nobukazu Yoshioka Oct. 27 1993
 */

#include <stdio.h>
#include <stdlib.h>
#include "c-ari.h"
#include "bitio.h"

extern long u;
extern int w;

long X, Y;			/* 区間 */
long L;				/* 復号するビット数 */
long Code[CODE_SIZE];		/* 文字の累積頻度 */

/**********************************************************
 *	Header部から出現頻度を復元する
 */
int start_input_bits(FILE * input_fp)
{
    int i;

    for (i = 0; i <= CODE_SIZE; i++) {	/* Codeの初期化 */
	Code[i] = 0;
    }

    for (i = 0; i < EOF_symbol; i++) {
	int c;
	c = fgetc(input_fp);
	if (c == EOF)
	    return -1;
	Code[i] = c;		/* 上位バイトから読み込む */
	Code[i] <<= 8;
	c = fgetc(input_fp);	/* 下位バイト */
	if (c == EOF)
	    return -1;
	Code[i] |= c;
    }
    Code[EOF_symbol] = 1;	/* EOF_symbolの出現頻度は常に1 */
    return 0;
}

/*
 *	累積頻度をCodeに設定しX,Yを初期化する
 */
int start_decoding(FILE * input_fp)
{
    int i;

    if (make_Fi())		/* 累積頻度をCodeに設定 */
	return -1;

    Y = 1L;			/* 区間を初期化 */
    Y <<= w;
    X = 0L;
    X = (bit_input(input_fp) & 0x01);	/* wビットをXに代入 */
    for (i = 1; i < w; i++) {
	X <<= 1;
	X |= (bit_input(input_fp) & 0x01);
    }
    L = 0L;
    return 0;
}

/*
 *	F_{i-1} <= e < F_i を満たすインデックスiを返す
 *	２分検索を使っている.
 */
int search_index(long e)
{
    int idx;
    int start = 0, end = 9;

    while (start < end) {
	idx = (start + end) / 2;
	if (Code[idx] < e)
	    start = idx + 1;
	else
	    end = idx;
    }

    while (Code[idx] <= e)
	idx++;
    return idx;
}

/*
 *	input_fpからの符号を文字にして返す
 *	ファイルの最後はEOF_symbol記号で終わる.
 */
int decode_symbol(FILE * input_fp)
{
    long Fi_1, Fi;		/* F_{i-1}とF_i */
    long e, Z, V;
    int s, i, symbol, add;

    e = ((double) u * ((double) 2 * X + 1) - 1) / ((double) 2 * Y);
    symbol = search_index(e);	/* 対応する文字を求める */

    Fi_1 = (symbol == 0) ? 0 : Code[symbol - 1];
    Fi = Code[symbol];
    Z = ((double) 2 * Y * Fi_1 + u) / ((double) 2 * u);

    X -= Z;
    V = ((double) 2 * Y * Fi + u) / ((double) 2 * u) - Z;
    if (V == 0) {
	fprintf(stderr, "fatal error!\n");
	exit(EXIT_FAILURE);
    }
    s = 0;
    while ((V << s) < (0x01L << w))
	s++;			/* 2^w <= V*2^s < 2^{w+1}を満たすsを求める */
    X <<= s;
    add = 0;			/* 次の符号語を作る */
    for (i = s; i > 0; i--) {
	add <<= 1;
	add |= (bit_input(input_fp) & 0x01);
    }
    X += add;
    Y = V << s;

    return symbol;
}

/******************************************************************
 *	MAIN PROGRAM FOR DECODING
 *	usage: decode input output
 */
int main(int argc, char *argv[])
{
    FILE *input_fp, *output_fp;

    if (argc != 3) {
	fprintf(stderr, "usage: decode input_file output_file\n");
	exit(EXIT_FAILURE);
    }
    input_fp = fopen(argv[1], "rb");	/* 読み込み用ファイル */
    if (input_fp == NULL) {
	fprintf(stderr, "Cannot open %s!\n", argv[1]);
	exit(EXIT_FAILURE);
    }
    output_fp = fopen(argv[2], "wb");	/* 書き込み用ファイル */
    if (output_fp == NULL) {
	fprintf(stderr, "Cannot open %s!\n", argv[2]);
	exit(EXIT_FAILURE);
    }
    start_input_bits(input_fp);	/*　出現頻度の復元 */
    if (start_decoding(input_fp))	/* 初期設定 */
	exit(EXIT_FAILURE);

    for (;;) {
	int symbol;
	symbol = decode_symbol(input_fp);
	if (symbol == EOF_symbol)
	    break;
	fputc(symbol, output_fp);
    }
    fclose(input_fp);
    fclose(output_fp);
    return 0;
}

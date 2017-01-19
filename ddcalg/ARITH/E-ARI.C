/*
 *		算術圧縮 符号化ルーチン(Jones符号)
 *			(c) Nobukazu Yoshioka Oct. 27 1993
 */

#include <stdio.h>
#include <stdlib.h>
#include "c-ari.h"
#include "bitio.h"

extern long u;
extern int w;

long X, Y;			/* 区間 */
long L;				/* 複号するビット数 */
int r = 0;			/* w+2ビット目からの1の数 */
long Code[CODE_SIZE];		/* 文字の累積頻度 */

/*********************************************************
 *	文字の出現頻度をカウントしCodeに代入する (１６ビットに丸められる)
 */
int start_model(FILE * input_fp)
{
    int shift_leaf(void);
    int c;

    for (c = 0; c <= CODE_SIZE; c++) {	/* Codeの初期化 */
	Code[c] = 0;
    }

    while ((c = fgetc(input_fp)) != EOF)
	Code[c]++;		/* 頻度のカウント */

    Code[EOF_symbol] = 1;	/* EOFの追加 */

    for (c = 0; c < CODE_SIZE; c++) {	/* １６ビットに丸めこむ */
	while ((unsigned long) (Code[c] & 0xFFFF) != Code[c]) {
	    shift_frq();
	}
    }
    return 0;
}

/********************************************************:
 *	すべての出現頻度を１ビット右にシフトする
 *	０以外のものは０にはならない
 */
int shift_frq(void)
{
    int i;

    for (i = 0; i < CODE_SIZE; i++)
	Code[i] = (Code[i] >> 1) | (Code[i] & 0x01);

    return 0;
}

/*****************************************************************
 *	headerをout_fpに出力する
 *	出現頻度を16ビットで書き込んでいる
 *	あらかじめ丸め込んでおくこと
 */
int start_output_bits(FILE * out_fp)
{
    int i;

    for (i = 0; i < EOF_symbol; i++) {	/* headerの出力 */
	/* 上位バイトから書き込む */
	fputc((unsigned char) ((Code[i] >> 8) & 0xff), out_fp);
	fputc((unsigned char) (Code[i] & 0xff), out_fp);
    }
    return 0;
}

/*
 *	累積頻度をCodeに設定しX,Yを初期化する
 */
int start_encoding(void)
{
    if (make_Fi())		/* 累積頻度をCodeに設定 */
	return -1;

    X = 0L, Y = 1L;		/* 区間を初期化 */
    Y <<= w;
    L = 0L;
    r = 0;
    return 0;
}

/*
 *	symbolを符号化して出力
 */

int encode_symbol(FILE * output_fp, int symbol)
{
    static int high_bit = -1;	/* 最上位ビット 0 or -1:存在しない */
    static bit_X = 0;		/* Xの桁数 */
    long V;
    long Fi_1, Fi;		/* F_{i-1}とF_i */
    int s;
    long Y2;			/* Y*Fi_1/uが入る */
    int shift;			/* shiftするビット数 */

    Fi_1 = (symbol == 0) ? 0 : Code[symbol - 1];
    Fi = Code[symbol];

    V = ((double) 2 * Y * Fi + u) / ((double) 2 * u);
    Y2 = ((double) 2 * Y * Fi_1 + u) / ((double) 2 * u);

    V -= Y2;

    if (V == 0) {
	fprintf(stderr, "fatal error!:symbol = %x\n", symbol);
	exit(-1);
    }
    s = 0;
    while ((V << s) < (0x01L << w))
	s++;			/* 2^w <= V*2^s < 2^{w+1}を満たすsを求める */
    shift = s;
    X += Y2;

    if (bit_X == 0) {		/* Xにまだ値が設定されていない */
	X <<= 1;		/* w+1桁に合わせる */
	shift--;
	bit_X = w + 1;
	r = 0;
    } else if (bit(X, w + 2)) {	/* 繰が上がったか？ */
	bit_out(output_fp, 1);	/* 繰り上がったビット */
	bits_out(output_fp, r, 0);
	r = 0;
	high_bit = -1;		/* rビットより上位は出力済み */
    }
    for (; shift > 0; shift--) {/* sビットシフト */
	if ((bit(X, w + 1) == 0)) {
	    if (high_bit == 0)
		bit_out(output_fp, 0);	/* 最上位ビット */
	    bits_out(output_fp, r, 1);
	    high_bit = 0;
	    r = 0;
	} else
	    r++;
	X <<= 1;
    }

    X = bit_omit(X, w + 1);	/* w+1ビットでmaskする */
    L += s;
    Y = (V << s);

    return 0;
}

/******************************************************************
 *	1でビットをフラッシュする
 */
int bit_flush(FILE * fp)
{
    bit_out(fp, 0);
    for (; r > 0; r--)		/* 残りの出力 */
	bit_out(fp, 1);
    bit_out(fp, bit(X, w + 1));
    while (bit_out(fp, 1) != 0);/* 1でフラッシュ */
}

/******************************************************************
 *	MAIN PROGRAM FOR ENCODING
 *	usage: encode input output
 */
int main(int argc, char *argv[])
{
    FILE *input_fp, *output_fp;


    if (argc != 3) {
	fprintf(stderr, "usage: encode input_file output_file\n");
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
    start_model(input_fp);	/* 文字の頻度を計算する */
    start_output_bits(output_fp);	/* headerを出力 */

    if (start_encoding())	/* 初期設定 */
	exit(EXIT_FAILURE);

    rewind(input_fp);
    for (;;) {
	int symbol;
	symbol = getc(input_fp);
	if (symbol == EOF)
	    break;
	encode_symbol(output_fp, symbol);
    }
    encode_symbol(output_fp, EOF_symbol);
    bit_flush(output_fp);

    fclose(input_fp);
    fclose(output_fp);


    return 0;
}

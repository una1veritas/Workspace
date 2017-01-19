/*
 *	encode for Huffman
 *	使い方: encode 入力ファイル 出力ファイル
 *					(c) Nobukazu Yoshioka Sep. 24 1993
 */
#include <stdio.h>
#include <stdlib.h>
#include "c-huff.h"
#include "bitio.h"

NODE Nodes[LEAF_SIZE * 2];
int Node_size = LEAF_SIZE;
int Root = NIL;

/*********************************************************
 *	文字の出現頻度のカウント (１６ビットに丸められる)
 *	Huffman木の葉の部分を初期化する.
 */
int start_model(FILE * input_fp)
{
    int shift_leaf(void);
    int c;

    for (c = 0; c < LEAF_SIZE; c++) {	/* 葉の初期化 */
	Nodes[c].frq = 0;
	Nodes[c].left = Nodes[c].right = Nodes[c].up = NIL;
    }

    while ((c = fgetc(input_fp)) != EOF)
	Nodes[c].frq++;		/* 頻度のカウント */

    Nodes[EOF_symbol].frq = 1;

    for (c = 0; c < LEAF_SIZE; c++) {	/* １６ビットに丸めこむ */
	while ((unsigned long) (Nodes[c].frq & 0xFFFF) != Nodes[c].frq) {
	    shift_leaf();
	}
    }
    return 0;
}

/*********************************************************
 *	葉の頻度を１ビット右にシフトする
 *	０以外のものは０にはならない
 */
int shift_leaf(void)
{
    int i;

    for (i = 0; i < LEAF_SIZE; i++)
	Nodes[i].frq = (Nodes[i].frq >> 1) | (Nodes[i].frq & 0x01);

    return 0;
}

/*********************************************************
 *	headerをout_fpに出力する
 *	出現頻度を16ビットで書き込んでいる
 *	あらかじめ丸め込んでおくこと
 */
int start_output_bits(FILE * out_fp)
{
    int i;

    for (i = 0; i < EOF_symbol; i++) {	/* headerの出力 */
	/* 上位バイトから書き込む */
	fputc((unsigned char) ((Nodes[i].frq >> 8) & 0xff), out_fp);
	fputc((unsigned char) (Nodes[i].frq & 0xff), out_fp);
    }
    return 0;
}

/*********************************************************
 *	Huffman木の作成 作れなかった場合 0以外を返す.
 */
int start_encoding(void)
{
    if (make_tree()== NIL) {
	fprintf(stderr, "Error: Huffman tree was broken!\n");
	return -1;
    } else
	return 0;
}

/*********************************************************
 * 文字 cを符号化して out_fpに出力する
 * 枝leftを1, rightを0としている.
 * 葉からスキャンして根から出力する
 */
int encode_symbol(FILE * out_fp, int c)
{
    int up_node;
    int bp = 0;			/* 符号化したビット数 */
    int bit_code[LEAF_SIZE];	/* 符号化したビット */

    do {
	up_node = Nodes[c].up;	/* rootに向かってスキャン */
	bit_code[bp++] = (Nodes[up_node].left == c) ? 1 : 0;
	c = up_node;
    } while (up_node != Root);

    while (--bp >= 0) {		/* 逆順に出力 */
	bit_out(out_fp, bit_code[bp]);
    }
    return 0;
}

/*********************************************************
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
    if (start_encoding())	/* 木の作成 */
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
    flush(output_fp);
    fclose(input_fp);
    fclose(output_fp);
    return 0;
}

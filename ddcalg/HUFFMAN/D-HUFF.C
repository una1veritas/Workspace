/*
 *	decode for Huffman
 *	使い方: decode 入力ファイル 出力ファイル
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
 *	Header部からNodeを復元する
 */
int start_input_bits(FILE * input_fp)
{
    int i;

    for (i = 0; i < LEAF_SIZE; i++) {	/* 葉の初期化 */
	Nodes[i].frq = 0;
	Nodes[i].left = Nodes[i].right = Nodes[i].up = NIL;
    }

    for (i = 0; i < EOF_symbol; i++) {
	int c;
	c = fgetc(input_fp);
	if (c == EOF)
	    return -1;
	Nodes[i].frq = c;	/* 上位バイトから読み込む */
	Nodes[i].frq <<= 8;
	c = fgetc(input_fp);	/* 下位バイト */
	if (c == EOF)
	    return -1;
	Nodes[i].frq |= c;
    }
    Nodes[EOF_symbol].frq = 1;	/* EOF_symbolの出現頻度は常に1 */
    return 0;
}

/*********************************************************
 *	Huffman木の作成 作れなかった場合 0以外を返す.
 */

int start_decoding(void)
{
    if (make_tree()== NIL) {
	fprintf(stderr, "Error: Huffman tree was broken!\n");
	return -1;
    } else
	return 0;
}

/*********************************************************
 *	input_fpからの符号を文字にして返す
 *	ファイルの最後はEOF_symbolの符号で終わる.
 */
int decode_symbol(FILE * input_fp)
{
    int node = Root;


    while (!is_leaf(node)) {	/* Rootから根に向かってスキャン */
	int inbit;
	inbit = bit_in(input_fp);
	if (inbit == 0 && Nodes[node].right != NIL) {
	    node = Nodes[node].right;
	} else if (inbit == 1 && Nodes[node].left != NIL) {
	    node = Nodes[node].left;
	} else {
	    fprintf(stderr, "Error: Broken input file!\n");
	    exit(EXIT_FAILURE);
	}
    }
    return node;
}

/*********************************************************
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
    start_input_bits(input_fp);	/* Nodesの復元 */
    if (start_decoding())	/* 木の作成 */
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

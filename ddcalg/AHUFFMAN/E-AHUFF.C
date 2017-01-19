
/******************************************************
*     e-ahuff.c                            1993.10.23 *
*                         programmed by Kenichi Iwata *
* Japan Advanced Institute of Science and Technology  *
******************************************************/

#include <stdio.h>
#include "c-ahuff.h"
#include "bitio.h"

extern NODE nodes[ALPHABET_MAX];
extern int order_list[ALPHABET_MAX], lti, max_node, now;
extern FILE *input_fp, *output_fp;

/* 記号の符号化 */
void encode_symbol(int symbol)
{
    int posi, temp, count;
    long stack;

    /* count,stackの初期設定 */
    count = 0;
    stack = 0;
    /* 配列nodesにおける記号symbolの位置をposiに代入する */
    posi = search_alphabet(symbol);
    temp = posi;
    /* 符号語をbit_out()に出力する */
    do {
	count++;
	stack = stack << 1;
	if (nodes[nodes[posi].parent].right == posi) {
	    stack++;
	}
	posi = nodes[posi].parent;
    } while (nodes[posi].parent != ROOT);
    while (count > 0) {
	bit_out(output_fp, (int)(stack - ((stack >> 1) << 1)));
	stack = stack >> 1;
	count--;
    }
    /* 記号が未出現記号の場合、記号を出力する */
    if (temp == max_node) {
	put_symbol(symbol);
    }
}

/*  EOF記号の出力 */
void end_code_symbol(void)
{
    encode_symbol(EOF);
}

/* メイン・ルーチン */
void main(int argc, char *argv[])
{
    int symbol;

    start_files(argc, argv);	/* 入出力の準備 */
    start_coding();		/* 符号化アルゴリズムの初期化 */
    for (;;) {
	symbol = getc(input_fp);/* 記号を読み込む */
	if (symbol == EOF)
	    break;		/* EOF記号なら終了 */
	encode_symbol(symbol);	/* 記号を符号化 */
	update_model(symbol);	/* 情報源モデルの更新 */
    }
    end_code_symbol();		/* EOF記号の出力 */
    done_encoding();		/* 符号化の後始末 */
    done_files();		/* 入出力の後始末 */
    exit(0);
}

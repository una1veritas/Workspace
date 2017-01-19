
/******************************************************
*     d-ahuff.c                           1993.10.23  *
*                         programmed by Kenichi Iwata *
* Japan Advanced Institute of Science and Technology  *
******************************************************/

#include <stdio.h>
#include "c-ahuff.h"
#include "bitio.h"

extern NODE nodes[ALPHABET_MAX];
extern int order_list[ALPHABET_MAX], lti, max_node, now;
extern FILE *input_fp, *output_fp;

/* 符号語の復号化 */
int decode_symbol(void)
{
    int posi;

    posi = 0;

    while (nodes[posi].leaf == IS_NODE) {
	if (bit_in(input_fp) == 1) {
	    posi = nodes[posi].right;
	} else {
	    posi = nodes[posi].left;
	}
    }
    if (posi == max_node) {
	return (get_symbol());
    } else {
	return (nodes[posi].alphabet);
    }
}

/* メイン・ルーチン */
void main(int argc, char *argv[])
{
    int symbol;

    start_files(argc, argv);	/*  入出力の準備 */
    start_coding();		/* 復号化アルゴリズムの初期化 */
    for (;;) {
	symbol = decode_symbol();	/* 符号語を復号する */
	if (symbol == EOF)
	    break;		/* EOF記号なら終了 */
	putc(symbol, output_fp);
	update_model(symbol);	/* 情報源モデルの更新 */
    }
    done_files();		/* 入力の後始末 */
    exit(0);
}

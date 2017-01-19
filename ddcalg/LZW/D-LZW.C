/******************************************************
*     d-lzw.c             LZW decode       1993.10.26 *
*                         programmed by Kenichi Iwata *
* Japan Advanced Institute of Science and Technology  *
******************************************************/

#include <stdio.h>
#include "c-lzw.h"

extern NODE nodes[MAX_ENTRY_NUM];
extern FILE *input_fp, *output_fp;
extern int now, next, entry, buf, now_sym, parent_node;

/* ファイルより符号語を読み込み、復号を
行ない最長一致系列の参照番号を返す */
int get_number()
{
    static int entry_number = 256;
    int k, border, number, count;

    entry_number++;
    /* entry_number > MAX_ENTRY_NUMの調整 */
    if (entry_number > MAX_ENTRY_NUM) {
	entry_number--;
    }
    k = log_2(entry_number);
    border = power(k) - (entry_number);
    count = k - 1;
    number = 0;
    while (count > 0) {
	number = number << 1;
	number = number + bit_in(input_fp);
	count--;
    }
    /* 2^k-n以上の場合 */
    if (number >= border) {
	number = number << 1;
	number = number + bit_in(input_fp);
	number = number - border;
    }
    return (number);
}

/* 節点番号から得られる記号語列を
一記号語ごとに戻り値とする。*/
int decode_symbol()
{
    static int count = 0;
    static int stack[MAX_ENTRY_NUM];
    static int number = ROOT, pre_num = 0;

    if (count == 0) {
	/* ファイルより新たな最長一致系列の参照番号を得る */
	number = get_number();
	if (number < entry) {
	    pre_num = number;
	}
	/* 例外処理 */
	else if (number == entry) {
	    number = pre_num;
	    while (nodes[pre_num].parent != ROOT) {
		pre_num = nodes[pre_num].parent;
	    }
	    stack[count] = pre_num;
	    count++;
	    pre_num = entry;
	}
	/* 最長一致系列の辞書からの復号 */
	do {
	    stack[count] = nodes[number].alphabet;
	    number = nodes[number].parent;
	    count++;
	} while (number != ROOT);
	count--;
	return (stack[count]);
    } else {
	count--;
	return (stack[count]);
    }
}

/* EOF記号まで符号語の復号化と木の更新を行なう。*/
update_decode()
{
    int symbol;

    for (;;) {
	if (buf == ROOT) {
	    symbol = decode_symbol();	/* 符号語を読み込む */
	    if (symbol == EOF) {
		break;		/* EOF なら終了 */
	    }
	    putc((char) symbol, output_fp);
	} else {
	    symbol = buf;
	}
	if (search_symbol(symbol) == NEW) {
	    new_node(symbol);	/* 木の更新 */
	} else {
	    next_node();	/* 次の節点 */
	}
    }
}

/* メイン・ルーチン */
main(argc, argv)
int argc;
char **argv;
{
    start_files(argc, argv);	/* 入出力の準備 */
    start_coding();		/* 復号化アルゴリズムの初期化 */
    update_decode();		/* 符号語の復号化と木の更新 */
    done_files();		/* 入出力の後始末 */
    exit(0);
}

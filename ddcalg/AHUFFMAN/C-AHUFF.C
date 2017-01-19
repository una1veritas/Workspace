/******************************************************
*      c-ahuff.c                           1993.10.23 *
*                         programmed by Kenichi Iwata *
* Japan Advanced Institute of Science and Technology  *
******************************************************/

#include <stdio.h>
#include "c-ahuff.h"
#include "bitio.h"

NODE nodes[ALPHABET_MAX];
int order_list[ALPHABET_MAX], lti, max_node, now;
FILE *input_fp, *output_fp;

/*  入出力ファイルの準備 */
void start_files(int argc, char *argv[])
{

    /* アーギュメントのチェック */
    if (argc != 3) {
	printf("Usage: %s from-file to-file\n", *argv);
	exit(1);
    }
    /* 入力ファイルを rb で開く */
    if ((input_fp = fopen(argv[1], "rb")) == NULL) {
	perror(argv[1]);
	exit(1);
    }
    /* 出力ファイルを wb で開く */
    if ((output_fp = fopen(argv[2], "wb")) == NULL) {
	perror(argv[2]);
	exit(1);
    }
}

/* 符号・復号化アルゴリズムの初期化 */
void start_coding(void)
{
    int i;

    for (i = 0; i < ALPHABET_MAX; i++) {
	nodes[i].alphabet = NIL;
	nodes[i].parent = NIL;
	nodes[i].left = NIL;
	nodes[i].right = NIL;
	nodes[i].order = NIL;
	nodes[i].leaf = IS_LEAF;
	nodes[i].weight = 0;
	order_list[i] = NIL;
    }
    nodes[0].parent = ROOT;
    nodes[0].order = 0;
    order_list[0] = 0;
    lti = 0;
    max_node = 0;
    now = 0;
    /* EOFを木に登録する */
    update_model(EOF);
}

/* 木の入れ換えを行なう。*/
void change_order(int n1, int n2)
{
    int temp;

    if (nodes[nodes[n1].parent].left == n1) {
	if (nodes[nodes[n2].parent].left == n2) {
	    nodes[nodes[n1].parent].left = n2;
	    nodes[nodes[n2].parent].left = n1;
	} else if (nodes[nodes[n2].parent].right == n2) {
	    nodes[nodes[n1].parent].left = n2;
	    nodes[nodes[n2].parent].right = n1;
	}
    } else if (nodes[nodes[n1].parent].right == n1) {
	if (nodes[nodes[n2].parent].left == n2) {
	    nodes[nodes[n1].parent].right = n2;
	    nodes[nodes[n2].parent].left = n1;
	} else if (nodes[nodes[n2].parent].right == n2) {
	    nodes[nodes[n1].parent].right = n2;
	    nodes[nodes[n2].parent].right = n1;
	}
    } else {
	printf("singuler\n");
	exit(0);
    }
    temp = order_list[nodes[n1].order];	/* order list の入れ換え */
    order_list[nodes[n1].order] = order_list[nodes[n2].order];
    order_list[nodes[n2].order] = temp;
    temp = nodes[n1].parent;
    nodes[n1].parent = nodes[n2].parent;
    nodes[n2].parent = temp;
    temp = nodes[n1].order;
    nodes[n1].order = nodes[n2].order;
    nodes[n2].order = temp;
}

/*
引数 symbol の対応する葉の配列 nodes の位置を返す。
symbol に対応する葉がない場合は、0-節点の配列 nodes の位置
を表す max_node を返す。
*/
int search_alphabet(int symbol)
{
    int i;

    for (i = 0; i < max_node; i++)
	if (nodes[i].alphabet == symbol) {
	    return (i);
	}
    return (max_node);
}

/*
0-節点に対して、新たに2つの重み0の子を付け加え、そのうちの
左の葉を新たな0-節点とし、右の葉に記号語symbolを割り当て、
木の節点番号を付け直し、親の節点に移動する。また、lti に
いま作った右側の葉の番号を代入する。
*/
void zero_node(int symbol)
{
    int i, zero_node, sym_node;

    zero_node = max_node + 2;	/* 0-節点の配列の位置 */
    sym_node = max_node + 1;	/* symbol-節点の配列の位置 */
    nodes[max_node].left = zero_node;
    nodes[max_node].right = sym_node;
    nodes[max_node].leaf = IS_NODE;	/* 内部節点となる */
    nodes[zero_node].parent = max_node;
    nodes[zero_node].order = 0;
    nodes[sym_node].alphabet = symbol;
    nodes[sym_node].parent = max_node;
    nodes[sym_node].order = 1;
    for (i = 0; i < max_node + 1; i++)	/* 節点の番号の付け直し */
	nodes[i].order = nodes[i].order + 2;
    for (i = max_node; i > -1; i--)
	order_list[i + 2] = order_list[i];
    order_list[0] = zero_node;
    order_list[1] = sym_node;
    now = max_node;
    lti = sym_node;		/* lti の処理 */
    max_node = max_node + 2;
}

/*
現在の葉をその葉の含まれるブロックの先頭節点と入れ換える。
もし、現在の節点が0-節点の兄弟ならば、lti に現在の葉の配
列での位置を代入した後、現在の葉の親の節点に移動する。
*/
void search_head(void)
{
    int next, temp;

    temp = now;
    next = order_list[nodes[now].order + 1];

    while (nodes[temp].weight == nodes[next].weight
	   && nodes[temp].leaf == nodes[next].leaf) {
	next = order_list[nodes[temp].order + 2];
	temp = order_list[nodes[temp].order + 1];
    }
    if (now != temp) {
	change_order(now, temp);/* 節点の入れ換え */
    }
    if ((nodes[now].weight == 0 || nodes[now].weight == 1)
	&& nodes[now].parent != ROOT) {
	lti = now;
	now = nodes[now].parent;
    }
}

/*
現在の節点が葉であり、同一の重みを有する内部節点が
存在する場合、同一の重みを有する内部節点内のブロッ
クの先頭節点に現在の葉をスライドさせる。
*/
void search_inner(void)
{
    int next;

    next = order_list[nodes[now].order + 1];

    while (nodes[now].weight == nodes[next].weight
	   && nodes[next].leaf == IS_NODE
	   && nodes[next].parent != ROOT) {
	change_order(now, next);
	next = order_list[nodes[now].order + 1];
    }
    nodes[now].weight = nodes[now].weight + 1;
    now = nodes[now].parent;
}

/*
現在の節点が内部節点であり、かつ重みが１大きい葉が存在する場合
重みが一つだけ大きい葉のブロックの先頭節点に現在の節点を
スライドさせる。その後、現在の節点の重みを１増した後、
元の節点（入れ換える前の節点）の親 pre_parent に移動する。
*/
void check_weight_leaf(void)
{
    int next, weighter, pre_parent;

    pre_parent = nodes[now].parent;
    next = order_list[nodes[now].order + 1];
    weighter = nodes[now].weight + 1;

    while (weighter == nodes[next].weight
	   && nodes[next].leaf == IS_LEAF) {
	change_order(now, next);
	next = order_list[nodes[now].order + 1];
    }
    nodes[now].weight = nodes[now].weight + 1;
    now = pre_parent;
}

/*
現在の節点の状態により、節点のスライドと
重みのインクリメントを行なう。
*/
void slide_increment(void)
{
    if (nodes[now].leaf == IS_LEAF) {
	search_inner();
    } else if (nodes[now].leaf == IS_NODE) {
	check_weight_leaf();
    }
}

/*
V アルゴリズムを用いた、symbolによる、木の更新
*/
void update_model(int symbol)
{
    lti = 0;

    /* now に配列 nodes のsymbolの位置を代入*/
    now = search_alphabet(symbol);
    /* symbolが未出現の文字の場合 */
    if (now == max_node) {
	zero_node(symbol);
    }
    /* symbolに対応する記号がすでに葉に存在する場合 */
    else {
	search_head();
    }
    /* 節点がROOTになるまで、slide_increment()を繰り返す。*/
    while (now != 0) {
	slide_increment();
    }
    /* ROOT節点の重みのインクリメント */
    nodes[0].weight = nodes[0].weight + 1;
    /* lti の処理 */
    if (lti != 0) {
	now = lti;
	slide_increment();
    }
}

/* 記号symbolを１ビット毎に分け、bit_outに送る */
void put_symbol(int symbol)
{
    int bit;
    int i;

    bit = 0;

    for (i = 7; i > -1; i--) {
	bit = symbol >> i;
	symbol = symbol - (bit << i);
	bit_out(output_fp, bit);
    }
}

/* 記号symbolをファイルから取り込む */
int get_symbol(void)
{
    int i, bit, byte;

    byte = 0;

    for (i = 0; i < 8; i++) {
	bit = bit_in(input_fp);
	byte = bit + (byte << 1);
    }
    return (byte);
}

/* 符号化の後始末 */
void done_encoding(void)
{
    flush(output_fp);
}

/* 入出力の後始末 */
void done_files(void)
{
    fclose(input_fp);
    fclose(output_fp);
}

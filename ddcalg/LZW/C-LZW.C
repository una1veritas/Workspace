/******************************************************
*     c-lzw.c                              1993.10.26 *
*                         programmed by Kenichi Iwata *
* Japan Advanced Institute of Science and Technology  *
******************************************************/

#include <stdio.h>
#include "c-lzw.h"

NODE nodes[MAX_ENTRY_NUM];
FILE *input_fp, *output_fp;
int  now,next,entry,buf,now_sym,parent_node;

/*  入出力ファイルの準備 */
void start_files(int argc,char *argv[])    
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

/* 引数に対するlogを求め、戻り値とする*/
int log_2(int temp)
{
    int count=0;

    while( temp != 0 ) {
	count++;
	temp=temp>>1;
    }
    return( count );
}

/* 2のk乗を求める k>0 */
int power(int temp)
{
    int ans;

    ans = 1;

    while(temp>0) {
	ans = 2*ans;
	temp--;
    }
    return( ans );
}

/* 符号・復号化アルゴリズムの初期化 */
void start_coding( void )
{
    int i;

    /* 元からの辞書 */
    for (i=0;i<256;i++) {
	nodes[i].alphabet	= i;
	nodes[i].parent 	= ROOT;
	nodes[i].brother	= i+1;
	nodes[i].child		= END;
    }
    /* EOF記号の辞書登録 */
    nodes[256].alphabet		= EOF;
    nodes[256].parent		= ROOT;
    nodes[256].brother		= END;
    nodes[256].child		= END;
   /* 新たな辞書登録のための領域の初期化 */
    for (i=257;i<MAX_ENTRY_NUM;i++) {
	nodes[i].alphabet	= NIL;
	nodes[i].parent		= NIL;
	nodes[i].brother	= NIL;
	nodes[i].child		= NIL;
    }
    now		= 0;
    entry	= 257; 		/* すでに辞書に登録されている記号語数 */
    buf		= ROOT;
    next 	= ROOT;
    now_sym	= ROOT;
    parent_node	= ROOT;
}

/* 木に新たな節点を付け加える */
void new_node(int symbol )
{
    if ( entry < MAX_ENTRY_NUM ) {
	nodes[entry].alphabet = symbol;
	nodes[entry].parent   = parent_node;
	nodes[entry].brother  = END;
	nodes[entry].child    = END;
	/* 新たな節点を木に対し横に増やす場合 */
	if ( next == YOKO ) {		
    	    nodes[now].brother = entry;
	}
	/* 新たな節点を木に対し縦に増やす場合 */
	if ( next == TATE ) {
	    nodes[parent_node].child = entry;
	}
	entry++;	/* 登録された記号語数をインクリメントする */
    }
    now 	= 0;
    parent_node	= ROOT;
    now_sym 	= ROOT;
    buf 	= symbol;
}

/* 次の節点に移動する */
void next_node(void)
{
    parent_node = now;
    buf      	= ROOT;
    now 	= nodes[now].child;
}

/* 配列 nodes[i].alphabet == symbol
で且つ条件を満たす。i を返す。 */
int search_symbol(int symbol)
{
    if( now == END ) {
	next = TATE;
	return( NEW );
    }
    while( nodes[now].alphabet != symbol ) {
	if ( nodes[now].brother == END ) {
	    next = YOKO;
	    return( NEW );
	}
	now = nodes[now].brother;
    }
    now_sym = now;
    return( now );
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

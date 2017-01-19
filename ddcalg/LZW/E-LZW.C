/******************************************************
*     e-lzw.c             LZW encode       1993.10.26 *
*                         programmed by Kenichi Iwata *
* Japan Advanced Institute of Science and Technology  *
******************************************************/

#include <stdio.h>
#include "c-lzw.h"
#include "bitio.h"

extern NODE nodes[MAX_ENTRY_NUM];
extern FILE *input_fp, *output_fp;
extern int  now,next,entry,buf,now_sym,parent_node;

/* 引数の値をCBT符号で符号化する */
void encode_symbol_cbt(int number)
{
    int i,border,count,surplus,out_code[20];

    i		= log_2(entry);
    border 	= power(i)-(entry);
    /* 0<=n<2^[log_2 entry]-entry の場合 */
    if( number < border ) {
	count	= i-1;
    }
    /* 2^[log_2 entry]-entry<=n<entry の場合 */
    else {
	count 	= i;
	number 	= number + border;
    }
    /* CBT符号の作成 */
    for(i=0;i<count;i++) {
	out_code[i] = 0;
	surplus  = number-(( number>>1)<<1);
	number = number>>1;
	out_code[i] = surplus;
    }
    /* CBT符号のファイルへの出力 */
    for(i=count-1;i>-1;i--) {
	bit_out( output_fp,out_code[i] );
    }
}

/* EOFまで記号語を符号化と木の更新を行なう。*/
void update_encode(void)
{
    int symbol;

    for(;;) {
	if ( buf == ROOT ) {
	    symbol = getc(input_fp); 	/* 記号語を読み込む */
	    if (symbol == EOF) {
		encode_symbol_cbt( now_sym );
		break; 	/* EOF なら終了 */
	    }
	}
	else { 
	    symbol = buf;
	}
	if ( search_symbol(symbol) == NEW ) {	
	    encode_symbol_cbt( now_sym );
	    new_node( symbol );		/* 木の更新 */
	}
	else {
	    next_node();		/* 次の節点 */
	}
    }
}	     

/*  EOF記号の出力 */
void end_code_symbol(void)
{
    entry++;
    encode_symbol_cbt(256); /* nodes[256].alphabet == EOF */
}

/* メイン・ルーチン */
void main(int argc,char *argv[]) 
{
	start_files(argc,argv);    	/* 出力の準備 */
	start_coding();		   	/* 符号化アルゴリズムの初期化 */
	update_encode(); 	  	/* 記号語の符号化と木の更新 */
	end_code_symbol();		/* EOF記号の出力 */
	done_encoding();		/* 符号化の後始末 */
	done_files();			/* 入出力の後始末 */
	exit(0);
}

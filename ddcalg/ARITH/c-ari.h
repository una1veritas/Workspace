/*
 *	共通部分 定数の定義など
 *			(c) Nobukazu Yoshioka Oct. 27 1993
 */

#define CODE_SIZE	257	/* 読み込む文字の種類 + 1 */
#define EOF_symbol	256
#if !defined(EXIT_FAILURE)
#define EXIT_FAILURE (-1)
#endif

int make_Fi(void);
int bit(long x, int m);
long bit_omit(long x, int m);
int bits_out(FILE * fp, int n, int bit);
int bit_input(FILE * fp);

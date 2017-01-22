/*
 *	共通部分 定数の定義など
 *			(c) Nobukazu Yoshioka Sep. 24 1993
 */

#define LEAF_SIZE	257	/* 読み込む文字の種類 + 1 */
#define EOF_symbol	(LEAF_SIZE - 1)
#define NIL	(-1)
#if !defined(EXIT_FAILURE)
#define EXIT_FAILURE (-1)
#endif

typedef struct {
    long frq;
    int up;
    int left;
    int right;
} NODE;

int make_tree(void);
int tree_print(int node, int tab);
int print_leaf(void);
int is_leaf(int node);



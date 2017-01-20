/*
 *		共通部分 Huffman木の作成
 *			(c) Nobukazu Yoshioka Sep. 24 1993
 */

#include <stdio.h>
#include <stdlib.h>
#include "c-huff.h"

extern NODE Nodes[];
extern int Node_size;
extern int Root;

/*********************************************************
 *	Huffman 木を作り Rootを設定する
 *	作れなかったらNILをかえす.
 */

int make_tree(void) {
	int search_least(void);
	int left, right;

	left = search_least();
	while (left >= 0) {
		Nodes[left].up = Node_size; /* 新しいnodeの作成 */
		/* 上の行はsearch_least()が呼ばれる前に処理すること */
		/* さもないともう一度同じノードを返してしまう */
		Nodes[Node_size].up = NIL;
		Nodes[Node_size].left = left;
		Nodes[Node_size].right = NIL;

		right = search_least();
		if (right < 0) {
			Root = left; /* １番頻度の多いnodeがRootとなる. */
			if (Nodes[Root].left == NIL && Nodes[Root].right == NIL) {
				Root = Node_size; /* 木がnodeのみの時(null file) */
				Node_size++; /* Rootと葉だけの木を作る */
			}
			return Root;
		}
		Nodes[Node_size].right = right;
		Nodes[Node_size].frq = Nodes[left].frq + Nodes[right].frq;
		Nodes[right].up = Node_size;
		Node_size++;
		left = search_least();
	}
	Root = NIL; /* ここには絶対来ないはず */
	return NIL;
}

/*********************************************************
 *	searchしていないcodeの中で1番少ない頻度のものへのインデックスを返す.
 *	ただし,頻度が0のnodeは返らない.
 */

int search_least(void) {
	int i, least = NIL;

	for (i = 0; i < Node_size; i++) {
		/* 親の無いnodeだけ検索 */
		if (Nodes[i].up == NIL && Nodes[i].frq > 0) {
			if (least == NIL || Nodes[least].frq > Nodes[i].frq)
				least = i;
		}
	}
	if (least == NIL) /* 木ができあがっている */
		return -1;

	return least;
}

/*********************************************************
 *	nodeが葉なら0以外を返す
 */
int is_leaf(int node) {
	if (Nodes[node].left == NIL && Nodes[node].left == NIL)
		return -1;
	else
		return 0;
}

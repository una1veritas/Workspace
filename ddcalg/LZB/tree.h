/**********************************
 *                                *
 * Binary Search Tree Prototype   *
 *                                *
 *               Kouichi Kitamura *
 *                                *
 **********************************/

struct node {
    unsigned long key1;		/* 先頭3文字のコード */
    unsigned char key2;		/* 残り1文字のコード */
    struct node *left;
    struct node *right;
    int buf_start_pos;
    int buf_end_pos;		/* もっとも最近加えられた文字の絶対バッファ位置 */
};

typedef struct node Tree;
extern struct node *talloc(void);
extern void insert(unsigned long x1, unsigned char x2, struct node **p, int *b_ps, int b_pe, int *ret, int *leng);
extern void delete(unsigned long x1, unsigned char x2, struct node **p, int b_p);

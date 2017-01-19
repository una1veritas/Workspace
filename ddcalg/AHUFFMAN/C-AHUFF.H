
/******************************************************
*     a_huff.h                             1993.10.23 *
*                         programmed by Kenichi Iwata *
* Japan Advanced Institute of Science and Technology  *
******************************************************/

#define ALPHABET_MAX	(256 + 1)*2 + 1
#define ROOT		-1
#define NIL		-2
#define IS_LEAF		0
#define IS_NODE		1

typedef struct node {
    int alphabet;		/* 節点が端点ならば, その端点の保持する記号語 */
    int parent;			/* 節点の親の節点を指す */
    int left;			/* 節点の左の子節点を指す */
    int right;			/* 節点の右の子節点を指す */
    int order;			/* 節点のハフマン木でのオーダー */
    int leaf;			/* 節点が端点であるか内点であるかを示す */
    long weight;		/* 節点が保持する記号語の出現頻度を示す */
} NODE;

void start_files(int argc, char *argv[]);
void start_coding(void);
void change_order(int n1, int n2);
int search_alphabet(int symbol);
void zero_node(int symbol);
void search_head(void);
void search_inner(void);
void check_weight_leaf(void);
void slide_increment(void);
void update_model(int symbol);
void put_symbol(int symbol);
int get_symbol(void);
void done_encoding(void);
void done_files(void);

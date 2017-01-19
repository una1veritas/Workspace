/******************************************************
*     c-lzw.h                              1993.10.26 *
*                         programmed by Kenichi Iwata *
* Japan Advanced Institute of Science and Technology  *
******************************************************/

#define MAX_ENTRY_NUM	4096
#define ROOT		-1
#define NEW		-2
#define END		-3
#define NIL		-4
#define	TATE		-5
#define YOKO		-6

typedef struct node{
    int	alphabet;	/* 節点の保持する記号語 */
    int parent;		/* 節点の親節点を指す */
    int brother;	/* 節点の次の兄弟節点を指す */
    int child;		/* 節点の子節点を指す */
} NODE;

void start_files(int argc,char *argv[]);
int log_2(int temp);
int power(int temp);
void start_coding( void );
void new_node(int symbol );
void next_node(void);
int search_symbol(int symbol);
void done_encoding(void);
void done_files(void);


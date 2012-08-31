#include <stdio.h>
#include <stdlib.h>

/* Data 型の定義 */
typedef int Data;

bool equals(Data x, Data y) {
	return x == y;
}

bool lessthan(Data x, Data y) {
	return x < y;
}

/* TNode の定義 */
struct TNode {
	Data element;
	TNode * left, * right;
	
	// 新しいノードを葉として作る.
	TNode(Data x) {
		element = x;
		left = NULL;
		right = NULL;
	}
	
	// ノードが葉かどうかを答える
	bool is_leaf() {
		return (left == NULL) && (right == NULL);
	}

	// ノードを根とみなし, 構造とともにデータすべてをファイルに印字する．
	void print(FILE * fp) {
		fprintf(fp, "%d", element);
		if ( (left == NULL) && (right == NULL) ) {
			return;
		}
		fprintf(fp, "(");
		if (left != NULL)
			left->print(fp);
		fprintf(fp, ", ");
		if (right != NULL)
			right->print(fp);
		fprintf(fp,")");
	}
};


/* TreeSet の定義 */
struct TreeSet {
	TNode * root;
	
	// 空の木を作る
	TreeSet() {
		root = NULL;
	}

	// データ x を探す
	TNode * find(Data x) {
		return *find(&root, x);
	}
	
	// **node を根とみなし, データ x を探索
	TNode ** find(TNode ** node, Data x) {
		while ( *node != NULL ) {
			if ( equals( (*node)->element, x) ) 
				return node;
			if ( lessthan(x, (*node)->element) ) 
				node = &(*node)->left;
			else
				node = &(*node)->right;
		}
		return node;
	}
	
  // データ x の追加を試みる. 
	TNode ** insert(Data x) {
		TNode ** node;
		node = find(&root, x);
		if ( *node == NULL )
			*node = new TNode(x);
		return node;
	}
    
  // データ x の削除を試みる. 
	TNode ** remove(Data x) {
	        TNode ** node, * temp;
		node = find(&root, x);
		if ( *node == NULL )  // 木が空の場合
			return node;
		if ( (*node)->is_leaf() ) {
		  // データを持つノードが葉の場合
			delete *node;
			*node = NULL;
			return node;
		}
		if ( ((*node)->left == NULL) || ((*node)->right == NULL) ) {
		  // データを持つノードに子が高々一つの場合
			if ( (*node)->left == NULL ) 
				temp = (*node)->right;
			else 
				temp = (*node)->left;
			delete *node;
			*node = temp;
			return node;
		} 
		// データを持つノードが子を左右とも持つ場合
		TNode ** sub;  // 削除するノードの身代わり用
		sub = &((*node)->left);
		// 左の子の最も右にある子孫を探す
		for ( ; (*sub)->right != NULL; sub = &((*sub)->right) ) ;
		temp = *sub;
		// 身代わりノードを飛ばす. 子はあっても左だけ
		*sub = (*sub)->left;
		// 身代わりノードに削除するノードの子をつなぐ
		temp->left = (*node)->left;
		temp->right = (*node)->right;
		*node = temp;	// 身代わりノードで置き換える
		return node;
	}
	
	void print(FILE * fp) {
		root->print(fp);
	}
};

/* テスト用プログラム */
int main(int argc, char * argv[]) {
	TreeSet set;
	int i;
	
	for (i = 1; i < argc; i++) {
      if (argv[i][0] == '-' ) 
         continue;
		set.insert(atoi(argv[i]));
      set.print(stdout);
      printf("\n");
   }
	
	for (i = 1; i < 10; i++) {
		if ( set.find(i) != NULL )
			printf("%d is included.\n", i);
		else 
			printf("%d is not included.\n", i);
	}
	
	for (i = 1; i < argc; i++) {
      if (argv[i][0] == '-' ) {
         set.remove(atoi(argv[i]+1));
         set.print(stdout);
         printf("\n");
      }
   }
	
}

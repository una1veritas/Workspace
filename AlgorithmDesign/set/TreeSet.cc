#include <stdio.h>
#include <stdlib.h>

/* datatype �^�̒�` */
typedef int datatype;

bool equals(datatype x, datatype y) {
	return x == y;
}

bool lessthan(datatype x, datatype y) {
	return x < y;
}

/* TreeNode �̒�` */
struct TreeNode {
	datatype element;
	TreeNode * left, * right;
	
	// �V�����m�[�h��t�Ƃ��č��.
	TreeNode(datatype x) {
		element = x;
		left = NULL;
		right = NULL;
	}
	
	// �m�[�h���t���ǂ����𓚂���
	bool is_leaf() {
		return (left == NULL) && (right == NULL);
	}

	// �m�[�h�����Ƃ݂Ȃ�, �\���ƂƂ��Ƀf�[�^���ׂĂ��t�@�C���Ɉ󎚂���D
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


/* TreeSet �̒�` */
struct TreeSet {
	TreeNode * root;
	
	// ��̖؂����
	TreeSet() {
		root = NULL;
	}

	// �f�[�^ x ��T��
	TreeNode * find(datatype x) {
		return *find(&root, x);
	}
	
	// **node �����Ƃ݂Ȃ�, �f�[�^ x ��T��
	TreeNode ** find(TreeNode ** node, datatype x) {
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
	
  // �f�[�^ x �̒ǉ������݂�. 
	TreeNode ** insert(datatype x) {
		TreeNode ** node;
		node = find(&root, x);
		if ( *node == NULL )
			*node = new TreeNode(x);
		return node;
	}
    
  // �f�[�^ x �̍폜�����݂�. 
	TreeNode ** remove(datatype x) {
	        TreeNode ** node, * temp;
		node = find(&root, x);
		if ( *node == NULL )  // �؂���̏ꍇ
			return node;
		if ( (*node)->is_leaf() ) {
		  // �f�[�^�����m�[�h���t�̏ꍇ
			delete *node;
			*node = NULL;
			return node;
		}
		if ( ((*node)->left == NULL) || ((*node)->right == NULL) ) {
		  // �f�[�^�����m�[�h�Ɏq�����X��̏ꍇ
			if ( (*node)->left == NULL ) 
				temp = (*node)->right;
			else 
				temp = (*node)->left;
			delete *node;
			*node = temp;
			return node;
		} 
		// �f�[�^�����m�[�h���q�����E�Ƃ����ꍇ
		TreeNode ** sub;  // �폜����m�[�h�̐g����p
		sub = &((*node)->left);
		// ���̎q�̍ł��E�ɂ���q����T��
		for ( ; (*sub)->right != NULL; sub = &((*sub)->right) ) ;
		temp = *sub;
		// �g����m�[�h���΂�. �q�͂����Ă�������
		*sub = (*sub)->left;
		// �g����m�[�h�ɍ폜����m�[�h�̎q���Ȃ�
		temp->left = (*node)->left;
		temp->right = (*node)->right;
		*node = temp;	// �g����m�[�h�Œu��������
		return node;
	}
	
	void print(FILE * fp) {
		root->print(fp);
	}
};

/* �e�X�g�p�v���O���� */
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

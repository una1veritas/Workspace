#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* definitions for `datatype.' */
typedef char datatype[256];

int equals(datatype a, datatype b) {
  return ! strcmp(a,b);
}

int lessthan(datatype a, datatype b) {
  return strcmp(a,b) < 0;
}

void putinto(datatype a, datatype b) {
  strcpy(b, a);
}

void print(datatype a) {
  printf("%s", a);
}


typedef struct Node {
  datatype data;
  struct Node * left, * right;
} Node;

Node * new_node(datatype d) {
  Node * ptr;
  
  ptr = (Node *) malloc(sizeof(Node));
  ptr->left  = NULL;
  ptr->right = NULL;
  putinto(d, ptr->data);
  return ptr;
}

Node ** find(Node ** hdl, datatype d) {
  while (*hdl != NULL) {
    if ( equals(d, (*hdl)->data) ) 
      return hdl;
    if ( lessthan(d, (*hdl)->data) ) 
      hdl = & (*hdl)->left;
    else
      hdl = & (*hdl)->right;
  }
  return hdl;
}

void Tree_initialize(Node ** hdl) {
  *hdl = NULL;
}

void Tree_add(Node ** hdl, datatype d) {
  hdl = find(hdl, d);
  if ( *hdl != NULL ) {
    return;
  }
  *hdl = new_node(d);
  return;
}

Node * extract(Node ** hdl) {
  Node * ptr;

  while ( (*hdl)->right != NULL ) {
    hdl = & (*hdl)->right;
  }
  ptr = *hdl;
  *hdl = (*hdl)->left;
  return ptr;
}

void Tree_delete(Node ** hdl, datatype d) {
  Node * ptr;

  hdl = find(hdl, d);
  if ( *hdl == NULL ) {
    return;
  }
  if ( (*hdl)->left == NULL && (*hdl)->right == NULL ) {
    free(*hdl); // the node (**hdl) is happened to be leaf.
    return;
  }
  if ( (*hdl)->left == NULL ) {
    ptr = *hdl;
    *hdl = ptr->right;
    free(ptr);
    return;
  } else if ( (*hdl)->right == NULL ) {
    ptr = *hdl;
    *hdl = ptr->left;
    free(ptr);
    return;
  }
  ptr = extract( & (*hdl)->left );
  ptr->left = (*hdl)->left;
  ptr->right = (*hdl)->right;
  free(*hdl);
  *hdl = ptr;
  return;
}


void Tree_print(Node ** hdl) {
  if ( *hdl == NULL ) 
    return;
  print((*hdl)->data);
  if ( (*hdl)->left == NULL && (*hdl)->right == NULL )
     return;
  printf("(");
  Tree_print( & (*hdl)->left );
  printf(",");
  Tree_print( & (*hdl)->right );
  printf(")");
}

int main(int argc, char * argv[]) {
  Node ** tree;
  int i;

  Tree_initialize(tree);

  for(i = 1; i < argc && *(argv[i]) != '-'; i++) {
    Tree_add(tree, argv[i]);
    Tree_print(tree);
    printf("\n");
  }

  for(i++; i < argc; i++) {
    Tree_delete(tree, argv[i]);
    Tree_print(tree);
    printf("\n");
  }

  return 0;
}

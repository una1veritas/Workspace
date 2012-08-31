/*
 * {1,...,N} の部分集合の数え上げ
 *
 * enumeration1 --- 再帰法による k 要素部分集合の数え上げを 1 <= k <= N に
 * ついて行う. 
 * enumeration2 --- ビットならびによる部分集合の表現とインクリメントによる
 * 部分集合の数え上げを行う. ただしビット列として unsigned long を使うので
 * 総要素数は 31 以下とする. 
 */

#include <stdio.h>
#include <stdlib.h>

void recursiveEnum1(int s, int e, int i, int sub[]) {
  int j, k;

  if (i == 0) {      /* 目標要素数に達したら */
    printf("{ ");    /* sub[] の内容をプリントして終了 */
    for(k = 1; k <= e; k++) {
      if (sub[k] != 0)
	printf("%d, ", k);
    }
    printf(" }\n");
    return;
  }
  for (j = s; j <= e; j++) {
    sub[j] = 1;        /* 要素として j を追加しさらに再帰呼び出し */
    recursiveEnum1(j+1, e, i-1, sub);
    sub[j] = 0;        /* sub を関数が呼び出される前の状態にもどす */
  }
}

void enumeration1(int n) {
  int i, j;
  int sub[n+1];     /* sub[i] != 0 で要素 i はあり, == 0 でなし */

  for (i = 1; i <= n; i++)      /* sub を空集合に初期化 */
    sub[i] = 0;
  for (i = 0; i <= n; i++) {
    recursiveEnum1(1,n,i,sub);  /* 要素数 i の部分集合を枚挙 */
  }
}

void enumeration2(int n) {
  unsigned long s;
  int i, j;
  //int sub[n+1];                /* sub[i] != 0 で要素 i はあり, == 0 でなし */
  
  if (n > 31) {   /* unsigned int では 32 ビットまでなのでチェック */
    printf("Overflow... Stopped.\n");
    return;
  }

  for (s = 0; s < (1 << n); s++) {  /* n 個の 1 が並んだら枚挙の最後 */
    printf("{");
    for (i = 0, j = 1; i < n; i++, j  = j << 1 ) {       /* iビット目は i+1 番目の要素に対応 */
      //      sub[i+1] = (s >> i) % 2;      /* s の i 番ビットをチェック */
      //      if ( sub[i+1] )             /* 表示用 */
      if ( s & j ) 
	printf(" %d,", i+1);
    }
    printf("  }\n");
  }
}

void recursiveEnum2(int p, int e, int sub[]) {
  int k;

  if (p > e) {       /* 要素をすべて検討したら */
    printf("{ ");    /* sub[] の内容をプリントして終了 */
    for(k = 1; k <= e; k++) {
      if (sub[k] != 0)
	printf("%d, ", k);
    }
    printf(" }\r");
    return;
  }
  sub[p] = 1;        /* 要素として p を追加する場合 */
  recursiveEnum2(p+1, e, sub);
  sub[p] = 0;        /* 要素として p を追加しない場合 */
  recursiveEnum2(p+1, e, sub);
}

void enumeration3(int n) {
  int i, j;
  int sub[n+1];     /* sub[i] != 0 で要素 i はあり, == 0 でなし */
  
  for (i = 1; i <= n; i++)    /* sub を空集合に初期化 */
    sub[i] = 0;
  recursiveEnum2(1,n,sub);
}

int main(int argc, char * argv[]) {
  int N = 6;

  N = atoi(argv[1]);
  
  enumeration3(N);

  printf("\n");
  return 0;
}

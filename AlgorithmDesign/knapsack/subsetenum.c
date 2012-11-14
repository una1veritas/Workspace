/*
 * {1,...,N} $B$NItJ,=89g$N?t$(>e$2(B
 *
 * enumeration1 --- $B:F5"K!$K$h$k(B k $BMWAGItJ,=89g$N?t$(>e$2$r(B 1 <= k <= N $B$K(B
 * $B$D$$$F9T$&(B. 
 * enumeration2 --- $B%S%C%H$J$i$S$K$h$kItJ,=89g$NI=8=$H%$%s%/%j%a%s%H$K$h$k(B
 * $BItJ,=89g$N?t$(>e$2$r9T$&(B. $B$?$@$7%S%C%HNs$H$7$F(B unsigned long $B$r;H$&$N$G(B
 * $BAmMWAG?t$O(B 31 $B0J2<$H$9$k(B. 
 */

#include <stdio.h>
#include <stdlib.h>

void recursiveEnum1(int s, int e, int i, int sub[]) {
  int j, k;

  if (i == 0) {      /* $BL\I8MWAG?t$KC#$7$?$i(B */
    printf("{ ");    /* sub[] $B$NFbMF$r%W%j%s%H$7$F=*N;(B */
    for(k = 1; k <= e; k++) {
      if (sub[k] != 0)
	printf("%d, ", k);
    }
    printf(" }\n");
    return;
  }
  for (j = s; j <= e; j++) {
    sub[j] = 1;        /* $BMWAG$H$7$F(B j $B$rDI2C$7$5$i$K:F5"8F$S=P$7(B */
    recursiveEnum1(j+1, e, i-1, sub);
    sub[j] = 0;        /* sub $B$r4X?t$,8F$S=P$5$l$kA0$N>uBV$K$b$I$9(B */
  }
}

void enumeration1(int n) {
  int i, j;
  int sub[n+1];     /* sub[i] != 0 $B$GMWAG(B i $B$O$"$j(B, == 0 $B$G$J$7(B */

  for (i = 1; i <= n; i++)      /* sub $B$r6u=89g$K=i4|2=(B */
    sub[i] = 0;
  for (i = 0; i <= n; i++) {
    recursiveEnum1(1,n,i,sub);  /* $BMWAG?t(B i $B$NItJ,=89g$rKg5s(B */
  }
}

void enumeration2(int n) {
  unsigned long s;
  int i, j;
  //int sub[n+1];                /* sub[i] != 0 $B$GMWAG(B i $B$O$"$j(B, == 0 $B$G$J$7(B */
  
  if (n > 31) {   /* unsigned int $B$G$O(B 32 $B%S%C%H$^$G$J$N$G%A%'%C%/(B */
    printf("Overflow... Stopped.\n");
    return;
  }

  for (s = 0; s < (1 << n); s++) {  /* n $B8D$N(B 1 $B$,JB$s$@$iKg5s$N:G8e(B */
    printf("{");
    for (i = 0, j = 1; i < n; i++, j  = j << 1 ) {       /* i$B%S%C%HL\$O(B i+1 $BHVL\$NMWAG$KBP1~(B */
      //      sub[i+1] = (s >> i) % 2;      /* s $B$N(B i $BHV%S%C%H$r%A%'%C%/(B */
      //      if ( sub[i+1] )             /* $BI=<(MQ(B */
      if ( s & j ) 
	printf(" %d,", i+1);
    }
    printf("  }\n");
  }
}

void recursiveEnum2(int p, int e, int sub[]) {
  int k;

  if (p > e) {       /* $BMWAG$r$9$Y$F8!F$$7$?$i(B */
    printf("{ ");    /* sub[] $B$NFbMF$r%W%j%s%H$7$F=*N;(B */
    for(k = 1; k <= e; k++) {
      if (sub[k] != 0)
	printf("%d, ", k);
    }
    printf(" }\r");
    return;
  }
  sub[p] = 1;        /* $BMWAG$H$7$F(B p $B$rDI2C$9$k>l9g(B */
  recursiveEnum2(p+1, e, sub);
  sub[p] = 0;        /* $BMWAG$H$7$F(B p $B$rDI2C$7$J$$>l9g(B */
  recursiveEnum2(p+1, e, sub);
}

void enumeration3(int n) {
  int i, j;
  int sub[n+1];     /* sub[i] != 0 $B$GMWAG(B i $B$O$"$j(B, == 0 $B$G$J$7(B */
  
  for (i = 1; i <= n; i++)    /* sub $B$r6u=89g$K=i4|2=(B */
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

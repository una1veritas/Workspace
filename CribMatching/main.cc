#include <stdio.h>
#include <string.h>

#define max(x,y) (x>y?x:y)
#define min(x,y) (x<y?x:y)

int main(int argc, char * argv[]) {
  char * ctxt, * crib;
  int ctxtlen, criblen;
  int mc;

  if ( !(argc >= 2+1) ) 
    return 1;
  ctxt = argv[1];
  ctxtlen = strlen(ctxt);
  crib = argv[2];
  criblen = strlen(crib);
  
  for(int p = 1; p < ctxtlen+criblen; p++) {
    mc = 0;
    for(int n = 0; n < criblen-1; n++) printf(" "); printf("%s\n", ctxt);
    for(int n = 1; n < p; n++) printf(" "); printf("%s\n", crib);
    for(int chk = max(criblen-p, 0); chk-criblen+p < min(criblen, ctxtlen); chk++) {
    	printf("%d:%d ", chk, chk-criblen+p);
    }
    printf("\n%d\n",mc);
    printf("\n");
  }

  return 0;
}

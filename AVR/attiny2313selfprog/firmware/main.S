#include<avr/io.h>
 
 
.global main
 
main:
#define   temp r16 
ldi      ZL,LOW(PAGESIZE*2) 
ldi      ZH,HIGH(PAGESIZE*2) 
ldi      temp,(1<<PGERS)|(1<<SPMEN) 
out      SPMCR,temp      ;for ATMega162 
;out   SPMCSR,temp      ;for ATTiny2313 
spm 
rjmp      pc 
.org   PAGESIZE 
.dw      0,1,2,3,4,5,6,7

.end

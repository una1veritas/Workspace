100 REM Pi Calculator
110 p=2
120 s=-1
130 CLS:INPUT "N=";n
140 FOR i=1 TO n
150 q=p
160 p=p+s*2/(2*i+1)
170 IF FRAC(i/200)=0 THEN PRINT i;" PI=";p+q
180 s=-s
190 NEXT i
200 PRINT "I=";i
210 PRINT "PI=";p+q
220 BEEP 0.5
230 END

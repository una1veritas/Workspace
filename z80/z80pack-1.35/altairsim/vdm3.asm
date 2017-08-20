;
; TEST PROGRAM FOR THE VDM CHARACTER ROM, TAKEN FROM MANUAL
;
; ENTER CHARACTER TO DISPLAY VIA SENSE SWITCHES 0-6, 7 IS INVERSE VIDEO
;
        MVI  A,0                ;CLEAR A
        OUT  0C8H               ;OUTPUT VDM PORT
        IN   0FFH               ;INPUT SENSE SWITCHES
REDO:   MOV  B,A
        MVI  A,0D0H
        LXI  H,0CC00H           ;INITIALIZE SCREEN POINTER
MOVE:   MOV  M,B
        INX  H                  ;INCREMENT SCREEN POINTER
        CMP  H                  ;COMPARE SCREEN POINTER
        JNZ  MOVE
INPUT:  IN   0FFH               ;INPUT SENSE SWITCHES
        CMP  B                  ;COMPARE TO B
        JZ   INPUT
        JMP  REDO               ;LOOP

        END

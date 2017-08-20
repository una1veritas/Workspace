;
; TEST PROGRAM FOR THE VDM CHARACTER ROM, SHOW THE COMPLETE SET
;
        MVI  A,0                ;INIT SCREEN TO SHOW ALL 1024 CHARACTERS
        OUT  0C8H
        LXI  H,0CC00H           ;INIT SCREEN POINTER
        MVI  B,0
LOOP:   DCR  B                  ;COUNT DOWN
        MOV  M,B                ;PUT (B) ON SCREEN
        INX  H                  ;INCREMENT SCREEN POINTER
        MOV  A,H
        CPI  0D0H               ;COMPARE POINTER WITH END OF SCREEN
        JNZ  LOOP
        HLT                     ;DONE

        END

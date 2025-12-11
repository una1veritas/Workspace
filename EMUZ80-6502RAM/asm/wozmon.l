ca65 V2.19 - N/A
Main file   : ./wozmon.s
Current file: ./wozmon.s

000000r 1               ;  The WOZ Monitor for the Apple 1
000000r 1               ;  Written by Steve Wozniak in 1976
000000r 1               ;  Port to 6502 SBC by Jeff Tranter
000000r 1               
000000r 1               ; Page 0 Variables
000000r 1               
000000r 1               XAML            = $24           ;  Last "opened" location Low
000000r 1               XAMH            = $25           ;  Last "opened" location High
000000r 1               STL             = $26           ;  Store address Low
000000r 1               STH             = $27           ;  Store address High
000000r 1               L               = $28           ;  Hex value parsing Low
000000r 1               H               = $29           ;  Hex value parsing High
000000r 1               YSAV            = $2A           ;  Used to see if hex value is given
000000r 1               MODE            = $2B           ;  $00=XAM, $7F=STOR, $AE=BLOCK XAM
000000r 1               
000000r 1               
000000r 1               ; Other Variables
000000r 1               
000000r 1               IN              = $0200         ;  Input buffer to $027F
000000r 1               
000000r 1               ACIA            = $B000         ; 6850 ACIA
000000r 1               ACIAControl     = ACIA+0
000000r 1               ACIAStatus      = ACIA+0
000000r 1               ACIAData        = ACIA+1
000000r 1               
000000r 1                              .org $C000
00C000  1               
00C000  1  00 00 00 00                 .res $FEEF-*,0   ; Adjust program to end at $FFFF
00C004  1  00 00 00 00  
00C008  1  00 00 00 00  
00FEEF  1               
00FEEF  1                              .export RESET
00FEEF  1               
00FEEF  1  D8           RESET:          CLD             ; Clear decimal arithmetic mode.
00FEF0  1  58                           CLI
00FEF1  1  A0 7F                        LDY #$7F
00FEF3  1  A9 15                        LDA #$15        ; Set ACIA to 8N1 and divide by 16 clock
00FEF5  1  8D 00 B0                     STA ACIAControl
00FEF8  1               
00FEF8  1  C9 DF        NOTCR:          CMP #'_'+$80    ; "_"?
00FEFA  1  F0 13                        BEQ BACKSPACE   ; Yes.
00FEFC  1  C9 9B                        CMP #$9B        ; ESC?
00FEFE  1  F0 03                        BEQ ESCAPE      ; Yes.
00FF00  1  C8                           INY             ; Advance text index.
00FF01  1  10 0F                        BPL NEXTCHAR    ; Auto ESC if > 127.
00FF03  1  A9 DC        ESCAPE:         LDA #'\'+$80    ; "\".
00FF05  1  20 DE FF                     JSR ECHO        ; Output it.
00FF08  1  A9 8D        GETLINE:        LDA #$8D        ; CR.
00FF0A  1  20 DE FF                     JSR ECHO        ; Output it.
00FF0D  1  A0 01                        LDY #$01        ; Initialize text index.
00FF0F  1  88           BACKSPACE:      DEY             ; Back up text index.
00FF10  1  30 F6                        BMI GETLINE     ; Beyond start of line, reinitialize.
00FF12  1  AD 00 B0     NEXTCHAR:       LDA ACIAStatus  ; Key ready?
00FF15  1  29 01                        AND #$01
00FF17  1  C9 01                        CMP #$01
00FF19  1  D0 F7                        BNE NEXTCHAR    ; Loop until ready.
00FF1B  1  AD 01 B0                     LDA ACIAData    ; Load character
00FF1E  1  09 80                        ORA #$80        ; B7 should be ‘1’.
00FF20  1  99 00 02                     STA IN,Y        ; Add to text buffer.
00FF23  1  20 DE FF                     JSR ECHO        ; Display character.
00FF26  1  C9 8D                        CMP #$8D        ; CR?
00FF28  1  D0 CE                        BNE NOTCR       ; No.
00FF2A  1  A0 FF                        LDY #$FF        ; Reset text index.
00FF2C  1  A9 00                        LDA #$00        ; For XAM mode.
00FF2E  1  AA                           TAX             ; 0->X.
00FF2F  1  0A           SETSTOR:        ASL             ; Leaves $7B if setting STOR mode.
00FF30  1  85 2B        SETMODE:        STA MODE        ; $00=XAM, $7B=STOR, $AE=BLOCK XAM.
00FF32  1  C8           BLSKIP:         INY             ; Advance text index.
00FF33  1  B9 00 02     NEXTITEM:       LDA IN,Y        ; Get character.
00FF36  1  C9 8D                        CMP #$8D        ; CR?
00FF38  1  F0 CE                        BEQ GETLINE     ; Yes, done this line.
00FF3A  1  C9 AE                        CMP #'.'+$80    ; "."?
00FF3C  1  90 F4                        BCC BLSKIP      ; Skip delimiter.
00FF3E  1  F0 F0                        BEQ SETMODE     ; Set BLOCK XAM mode.
00FF40  1  C9 BA                        CMP #':'+$80    ; ":"?
00FF42  1  F0 EB                        BEQ SETSTOR     ; Yes. Set STOR mode.
00FF44  1  C9 D2                        CMP #'R'+$80    ; "R"?
00FF46  1  F0 3B                        BEQ RUN         ; Yes. Run user program.
00FF48  1  86 28                        STX L           ; $00->L.
00FF4A  1  86 29                        STX H           ;  and H.
00FF4C  1  84 2A                        STY YSAV        ; Save Y for comparison.
00FF4E  1  B9 00 02     NEXTHEX:        LDA IN,Y        ; Get character for hex test.
00FF51  1  49 B0                        EOR #$B0        ; Map digits to $0-9.
00FF53  1  C9 0A                        CMP #$0A        ; Digit?
00FF55  1  90 06                        BCC DIG         ; Yes.
00FF57  1  69 88                        ADC #$88        ; Map letter "A"-"F" to $FA-FF.
00FF59  1  C9 FA                        CMP #$FA        ; Hex letter?
00FF5B  1  90 11                        BCC NOTHEX      ; No, character not hex.
00FF5D  1  0A           DIG:            ASL
00FF5E  1  0A                           ASL             ; Hex digit to MSD of A.
00FF5F  1  0A                           ASL
00FF60  1  0A                           ASL
00FF61  1  A2 04                        LDX #$04        ; Shift count.
00FF63  1  0A           HEXSHIFT:       ASL             ; Hex digit left, MSB to carry.
00FF64  1  26 28                        ROL L           ; Rotate into LSD.
00FF66  1  26 29                        ROL H           ; Rotate into MSD’s.
00FF68  1  CA                           DEX             ; Done 4 shifts?
00FF69  1  D0 F8                        BNE HEXSHIFT    ; No, loop.
00FF6B  1  C8                           INY             ; Advance text index.
00FF6C  1  D0 E0                        BNE NEXTHEX     ; Always taken. Check next character for hex.
00FF6E  1  C4 2A        NOTHEX:         CPY YSAV        ; Check if L, H empty (no hex digits).
00FF70  1  F0 91                        BEQ ESCAPE      ; Yes, generate ESC sequence.
00FF72  1  24 2B                        BIT MODE        ; Test MODE byte.
00FF74  1  50 10                        BVC NOTSTOR     ; B6=0 STOR, 1 for XAM and BLOCK XAM
00FF76  1  A5 28                        LDA L           ; LSD’s of hex data.
00FF78  1  81 26                        STA (STL,X)     ; Store at current ‘store index’.
00FF7A  1  E6 26                        INC STL         ; Increment store index.
00FF7C  1  D0 B5                        BNE NEXTITEM    ; Get next item. (no carry).
00FF7E  1  E6 27                        INC STH         ; Add carry to ‘store index’ high order.
00FF80  1  4C 33 FF     TONEXTITEM:     JMP NEXTITEM    ; Get next command item.
00FF83  1  6C 24 00     RUN:            JMP (XAML)      ; Run at current XAM index.
00FF86  1  30 2B        NOTSTOR:        BMI XAMNEXT     ; B7=0 for XAM, 1 for BLOCK XAM.
00FF88  1  A2 02                        LDX #$02        ; Byte count.
00FF8A  1  B5 27        SETADR:         LDA L-1,X       ; Copy hex data to
00FF8C  1  95 25                        STA STL-1,X     ;  ‘store index’.
00FF8E  1  95 23                        STA XAML-1,X    ; And to ‘XAM index’.
00FF90  1  CA                           DEX             ; Next of 2 bytes.
00FF91  1  D0 F7                        BNE SETADR      ; Loop unless X=0.
00FF93  1  D0 14        NXTPRNT:        BNE PRDATA      ; NE means no address to print.
00FF95  1  A9 8D                        LDA #$8D        ; CR.
00FF97  1  20 DE FF                     JSR ECHO        ; Output it.
00FF9A  1  A5 25                        LDA XAMH        ; ‘Examine index’ high-order byte.
00FF9C  1  20 CB FF                     JSR PRBYTE      ; Output it in hex format.
00FF9F  1  A5 24                        LDA XAML        ; Low-order ‘examine index’ byte.
00FFA1  1  20 CB FF                     JSR PRBYTE      ; Output it in hex format.
00FFA4  1  A9 BA                        LDA #':'+$80    ; ":".
00FFA6  1  20 DE FF                     JSR ECHO        ; Output it.
00FFA9  1  A9 A0        PRDATA:         LDA #$A0        ; Blank.
00FFAB  1  20 DE FF                     JSR ECHO        ; Output it.
00FFAE  1  A1 24                        LDA (XAML,X)    ; Get data byte at ‘examine index’.
00FFB0  1  20 CB FF                     JSR PRBYTE      ; Output it in hex format.
00FFB3  1  86 2B        XAMNEXT:        STX MODE        ; 0->MODE (XAM mode).
00FFB5  1  A5 24                        LDA XAML
00FFB7  1  C5 28                        CMP L           ; Compare ‘examine index’ to hex data.
00FFB9  1  A5 25                        LDA XAMH
00FFBB  1  E5 29                        SBC H
00FFBD  1  B0 C1                        BCS TONEXTITEM  ; Not less, so no more data to output.
00FFBF  1  E6 24                        INC XAML
00FFC1  1  D0 02                        BNE MOD8CHK     ; Increment ‘examine index’.
00FFC3  1  E6 25                        INC XAMH
00FFC5  1  A5 24        MOD8CHK:        LDA XAML        ; Check low-order ‘examine index’ byte
00FFC7  1  29 07                        AND #$07        ;  For MOD 8=0
00FFC9  1  10 C8                        BPL NXTPRNT     ; Always taken.
00FFCB  1  48           PRBYTE:         PHA             ; Save A for LSD.
00FFCC  1  4A                           LSR
00FFCD  1  4A                           LSR
00FFCE  1  4A                           LSR             ; MSD to LSD position.
00FFCF  1  4A                           LSR
00FFD0  1  20 D4 FF                     JSR PRHEX       ; Output hex digit.
00FFD3  1  68                           PLA             ; Restore A.
00FFD4  1  29 0F        PRHEX:          AND #$0F        ; Mask LSD for hex print.
00FFD6  1  09 B0                        ORA #'0'+$80    ; Add "0".
00FFD8  1  C9 BA                        CMP #$BA        ; Digit?
00FFDA  1  90 02                        BCC ECHO        ; Yes, output it.
00FFDC  1  69 06                        ADC #$06        ; Add offset for letter.
00FFDE  1  48           ECHO:           PHA
00FFDF  1  AD 00 B0     ECHO1:          LDA ACIAStatus
00FFE2  1  29 02                        AND #$02
00FFE4  1  C9 02                        CMP #$02
00FFE6  1  D0 F7                        BNE ECHO1       ; No, wait for display.
00FFE8  1  68                           PLA
00FFE9  1  48                           PHA
00FFEA  1  29 7F                        AND #$7F        ; Clear B7
00FFEC  1  8D 01 B0                     STA ACIAData    ; Output character.
00FFEF  1  C9 0D                        CMP #$0D        ; CR?
00FFF1  1  D0 05                        BNE RET
00FFF3  1  A9 0A                        LDA #$0A        ; If so, send LF
00FFF5  1  20 DE FF                     JSR ECHO
00FFF8  1  68           RET:            PLA
00FFF9  1  60                           RTS             ; Return.
00FFFA  1               
00FFFA  1               ; Interrupt Vectors
00FFFA  1               
00FFFA  1  EF FE                        .WORD RESET     ; NMI
00FFFC  1  EF FE                        .WORD RESET     ; RESET
00FFFE  1  EF FE                        .WORD RESET     ; BRK/IRQ
00FFFE  1               

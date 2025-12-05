# 8086_NASCOM_BASIC

Nascom BASICはマイクロソフトが作成し、Grant Searleさんがサブセットに改編し、鈴木哲哉さんが移植されたコードを元に、奥江聡が8080/Z80から8086にコンバートしました。  

ファイルは原作者の宣言にしたがってご利用ください。  

ターゲット SBCV20 V20/8088  
アセンブラ Macro Assembler 1.42  

BASICのMONITOR命令で割り込みを禁止して0C000hへジャンプします。

# 8080/Z80 から 8086 への変換ルール

条件リターンがないので条件ジャンプに変更する  
16bitのINC/DECでフラグ変化するので命令前後でフラグの保持する  
PUSH AFを別レジスタでPOPされる場合の整合をとる  
```
——————————————————
レジスタ対応
AF	AX
A	AL
BC	CX
B	CH
C	CL
DE	DX
D	DH
E	DL
HL	BX
H	BH
L	BL
——————————————————
条件ジャンプ
JP C	JC		Carry
JP NC	JNC		Not Carry
JP Z	JZ		Zero
JP NZ	JNZ		Not Zero
JP PE	JPE		Parity Even
JP PO	JPO		Parity Odd
JP P	JNS		Plus
JP M	JS		Minus

	JO		Overflow
	JNO		Not Overflow
——————————————————
ADD	ADD
ADC	ADC
AND	AND

CALL	CALL
CP	CMP
CPL	NOT
CCF	CMC

DEC	DEC  *CAUTION 16bit flag
DI	CLI

EX	XCHG	[SP],HL = [SP],BX / HL,DE = BX,DX
EI	STI

HALT	HLT

INC	INC  *CAUTION 16bit flag
IN	IN

JR	JMP
JP	JMP

LD	MOV

OR	OR
OUT	OUT

POP	POP
PUSH	PUSH

RET	RET
RLA	RCL AL,1
RRA	RCR AL,1
RL	RCL x,1
RR	RCR x,1
RLCA	ROL AL,1
RRCA	ROR AL,1

SUB	SUB
SBC	SBB
SLA	SAL x,1
SRA	SAR x,1
SRL	SHR
SCF	STC

XOR	XOR
——————————————————
PUSH AF

LAHF
XCHG AH,AL
PUSH AX
XCHG AH,AL
——————————————————
POP AF

POP  AX
XCHG AH,AL
SAHF
——————————————————
INC HL,DE,BC

LAHF
INC BX,DX,CX
SAHF
——————————————————
DEC HL,DE,BC

LAHF
DEC BX,DX,CX
SAHF
——————————————————
LD A,(DE)

XCHG BX,DX
MOV AL,[BX]
XCHG BX,DX
——————————————————
LD A,(BC)

XCHG BX,CX
MOV AL,[BX]
XCHG BX,CX
——————————————————
EX (SP),HL

MOV BP,SP
XCHG [BP],BX
——————————————————
JP (HL)

PUSH BX
RET
——————————————————
```

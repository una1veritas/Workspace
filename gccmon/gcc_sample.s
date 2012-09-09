	.cpu arm7tdmi
	.fpu softvfp
	.eabi_attribute 20, 1
	.eabi_attribute 21, 1
	.eabi_attribute 23, 3
	.eabi_attribute 24, 1
	.eabi_attribute 25, 1
	.eabi_attribute 26, 1
	.eabi_attribute 30, 1
	.eabi_attribute 34, 0
	.eabi_attribute 18, 4
	.file	"gcc_sample.c"
	.text
	.align	2
	.global	CheckRecvData
	.type	CheckRecvData, %function
CheckRecvData:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	@ link register save eliminated.
	ldr	r3, .L2
	ldrb	r0, [r3, #20]	@ zero_extendqisi2
	and	r0, r0, #1
	bx	lr
.L3:
	.align	2
.L2:
	.word	-536821760
	.size	CheckRecvData, .-CheckRecvData
	.align	2
	.global	RecvData
	.type	RecvData, %function
RecvData:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	@ link register save eliminated.
	ldr	r2, .L7
.L5:
	ldrb	r3, [r2, #20]	@ zero_extendqisi2
	tst	r3, #1
	beq	.L5
	ldr	r3, .L7
	ldrb	r0, [r3, #0]	@ zero_extendqisi2
	bx	lr
.L8:
	.align	2
.L7:
	.word	-536821760
	.size	RecvData, .-RecvData
	.align	2
	.global	SendData
	.type	SendData, %function
SendData:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	@ link register save eliminated.
	ldr	r2, .L12
.L10:
	ldrb	r3, [r2, #20]	@ zero_extendqisi2
	tst	r3, #32
	beq	.L10
	ldr	r3, .L12
	strb	r0, [r3, #0]
	bx	lr
.L13:
	.align	2
.L12:
	.word	-536821760
	.size	SendData, .-SendData
	.align	2
	.global	IRQ_func
	.type	IRQ_func, %function
IRQ_func:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r3, lr}
	mvn	r3, #0
	ldr	r3, [r3, #-4095]
	tst	r3, #64
	beq	.L16
	ldr	r3, .L17
	ldrb	r3, [r3, #8]	@ zero_extendqisi2
	mov	r3, r3, lsr #1
	and	r3, r3, #7
	cmp	r3, #2
	bne	.L16
	bl	RecvData
	bl	SendData
.L16:
	ldmfd	sp!, {r3, lr}
	bx	lr
.L18:
	.align	2
.L17:
	.word	-536821760
	.size	IRQ_func, .-IRQ_func
	.align	2
	.global	CPU_Initialize
	.type	CPU_Initialize, %function
CPU_Initialize:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	@ link register save eliminated.
	ldr	r3, .L27
	ldr	r3, [r3, #136]
	tst	r3, #33554432
	beq	.L20
	ldr	r3, .L27
	mov	r2, #1
	str	r2, [r3, #128]
	mov	r2, #170
	str	r2, [r3, #140]
	mov	r2, #85
	str	r2, [r3, #140]
.L20:
	ldr	r3, .L27
	mov	r2, #0
	str	r2, [r3, #128]
	mov	r2, #170
	str	r2, [r3, #140]
	mov	r2, #85
	str	r2, [r3, #140]
	mov	r2, #33
	str	r2, [r3, #416]
	mov	r2, r3
.L21:
	ldr	r3, [r2, #416]
	tst	r3, #64
	beq	.L21
	ldr	r3, .L27
	mov	r0, #1
	str	r0, [r3, #268]
	ldr	r2, .L27+4
	str	r2, [r3, #132]
	mov	r1, #170
	str	r1, [r3, #140]
	mov	r2, #85
	str	r2, [r3, #140]
	str	r0, [r3, #128]
	str	r1, [r3, #140]
	str	r2, [r3, #140]
	mov	r2, #3
	str	r2, [r3, #260]
	mov	r2, #5
	str	r2, [r3, #264]
	mov	r2, r3
.L22:
	ldr	r3, [r2, #136]
	tst	r3, #67108864
	beq	.L22
	ldr	r3, .L27
	mov	r2, #3
	str	r2, [r3, #128]
	mov	r2, #170
	str	r2, [r3, #140]
	mov	r2, #85
	str	r2, [r3, #140]
	mov	r2, r3
.L23:
	ldr	r3, [r2, #136]
	tst	r3, #33554432
	beq	.L23
	bx	lr
.L28:
	.align	2
.L27:
	.word	-534790144
	.word	721039
	.size	CPU_Initialize, .-CPU_Initialize
	.align	2
	.global	putscon
	.type	putscon, %function
putscon:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r4, lr}
	mov	r4, r0
	ldrb	r0, [r0, #0]	@ zero_extendqisi2
	cmp	r0, #0
	beq	.L29
.L31:
	bl	SendData
	ldrb	r0, [r4, #1]!	@ zero_extendqisi2
	cmp	r0, #0
	bne	.L31
.L29:
	ldmfd	sp!, {r4, lr}
	bx	lr
	.size	putscon, .-putscon
	.align	2
	.global	getscon
	.type	getscon, %function
getscon:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r3, r4, r5, r6, r7, r8, r9, sl, fp, lr}
	mov	sl, r0
	mov	r6, r2
	mov	r3, #0
	strb	r3, [r0, #0]
	subs	r8, r1, #0
	blt	.L40
	mov	r7, r3
	mov	r5, r3
	mov	fp, #8
.L39:
	bl	RecvData
	mov	r4, r0
	cmp	r0, #3
	beq	.L41
	cmp	r0, #13
	moveq	r3, #0
	streqb	r3, [sl, r7]
	beq	.L35
.L36:
	cmp	r0, #8
	cmpne	r0, #127
	bne	.L37
	cmp	r5, #0
	beq	.L37
	cmp	r5, r7
	bne	.L37
	mov	r3, #0
	strb	r3, [sl, r5]
	sub	r7, r5, #1
	mov	r0, fp
	bl	SendData
	mov	r0, #32
	bl	SendData
	mov	r0, fp
	bl	SendData
	mov	r5, r7
.L37:
	cmp	r6, #1
	movne	r3, #0
	moveq	r3, #1
	cmp	r4, #114
	cmpeq	r6, #1
	beq	.L42
	cmp	r4, #110
	movne	r3, #0
	andeq	r3, r3, #1
	cmp	r3, #0
	bne	.L43
	and	r3, r4, #127
	sub	r3, r3, #32
	cmp	r3, #94
	bhi	.L38
	cmp	r5, r7
	bne	.L38
	strb	r4, [sl, r5]
	add	r9, r5, #1
	mov	r0, r4
	bl	SendData
	cmp	r5, r9
	movge	r7, r5
	movlt	r7, r9
	mov	r5, r9
.L38:
	cmp	r8, r5
	bge	.L39
	b	.L34
.L40:
	mov	r7, #0
	mov	r5, r7
.L34:
	mov	r3, #0
	strb	r3, [sl, r5]
	b	.L35
.L41:
	mvn	r7, #0
	b	.L35
.L42:
	mov	r7, #114
	b	.L35
.L43:
	mov	r7, #110
.L35:
	mov	r0, r7
	ldmfd	sp!, {r3, r4, r5, r6, r7, r8, r9, sl, fp, lr}
	bx	lr
	.size	getscon, .-getscon
	.align	2
	.global	paramer
	.type	paramer, %function
paramer:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r3, lr}
	ldr	r0, .L47
	bl	putscon
	ldmfd	sp!, {r3, lr}
	bx	lr
.L48:
	.align	2
.L47:
	.word	.LC0
	.size	paramer, .-paramer
	.align	2
	.global	movem
	.type	movem, %function
movem:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	@ link register save eliminated.
	ldr	r2, [r0, #0]
	ldr	r3, [r0, #4]
	ldr	r0, [r0, #8]
	cmp	r1, #0
	bne	.L50
	cmp	r0, #0
	bxeq	lr
	sub	r2, r2, #1
	add	r0, r3, r0
.L52:
	ldrb	r1, [r2, #1]!	@ zero_extendqisi2
	strb	r1, [r3], #1
	cmp	r3, r0
	bne	.L52
	bx	lr
.L50:
	cmp	r1, #1
	bne	.L53
	sub	r1, r0, #1
	cmp	r0, #0
	bxeq	lr
	sub	r2, r2, #2
.L54:
	ldrh	r0, [r2, #2]!
	strh	r0, [r3], #2	@ movhi
	sub	r1, r1, #1
	cmn	r1, #1
	bne	.L54
	bx	lr
.L53:
	cmp	r1, #2
	bxne	lr
	sub	r1, r0, #1
	cmp	r0, #0
	bxeq	lr
	sub	r2, r2, #4
.L55:
	ldr	r0, [r2, #4]!
	str	r0, [r3], #4
	sub	r1, r1, #1
	cmn	r1, #1
	bne	.L55
	bx	lr
	.size	movem, .-movem
	.align	2
	.global	dumpmt
	.type	dumpmt, %function
dumpmt:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r3, lr}
	ldr	r0, .L60
	bl	putscon
	ldmfd	sp!, {r3, lr}
	bx	lr
.L61:
	.align	2
.L60:
	.word	.LC1
	.size	dumpmt, .-dumpmt
	.align	2
	.global	cmpstr
	.type	cmpstr, %function
cmpstr:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	@ link register save eliminated.
	str	r4, [sp, #-4]!
	add	ip, r0, #250
.L64:
	ldrb	r3, [r1], #1	@ zero_extendqisi2
	ldrb	r2, [r0], #1	@ zero_extendqisi2
	cmp	r3, r2
	bhi	.L65
	bcc	.L66
	orrs	r4, r2, r3
	beq	.L67
	cmp	r3, #0
	cmpne	r2, #0
	beq	.L68
	cmp	r0, ip
	bne	.L64
	mov	r0, #1
	b	.L63
.L65:
	mov	r0, #1
	b	.L63
.L66:
	mvn	r0, #0
	b	.L63
.L67:
	mov	r0, #0
	b	.L63
.L68:
	mov	r0, #1
.L63:
	ldmfd	sp!, {r4}
	bx	lr
	.size	cmpstr, .-cmpstr
	.align	2
	.global	cpysstr
	.type	cpysstr, %function
cpysstr:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	@ link register save eliminated.
	str	r4, [sp, #-4]!
	cmp	r2, #0
	beq	.L74
	ldrb	ip, [r1, #0]	@ zero_extendqisi2
	tst	ip, #223
	beq	.L75
	add	r3, r0, #1
	add	r4, r0, r2
	mov	r2, #0
	b	.L72
.L73:
	ldrb	ip, [r1, r2]	@ zero_extendqisi2
	add	r3, r3, #1
	tst	ip, #223
	beq	.L71
.L72:
	mov	r0, r3
	strb	ip, [r3, #-1]
	add	r2, r2, #1
	cmp	r3, r4
	bne	.L73
	b	.L71
.L74:
	mov	r2, #0
	b	.L71
.L75:
	mov	r2, #0
.L71:
	mov	r3, #0
	strb	r3, [r0, #0]
	mov	r0, r2
	ldmfd	sp!, {r4}
	bx	lr
	.size	cpysstr, .-cpysstr
	.align	2
	.global	spacen
	.type	spacen, %function
spacen:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r3, r4, r5, lr}
	cmp	r0, #0
	beq	.L76
	mov	r4, r0
	mov	r5, #32
.L78:
	mov	r0, r5
	bl	SendData
	subs	r4, r4, #1
	bne	.L78
.L76:
	ldmfd	sp!, {r3, r4, r5, lr}
	bx	lr
	.size	spacen, .-spacen
	.align	2
	.global	crlf
	.type	crlf, %function
crlf:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r3, lr}
	mov	r0, #13
	bl	SendData
	mov	r0, #10
	bl	SendData
	ldmfd	sp!, {r3, lr}
	bx	lr
	.size	crlf, .-crlf
	.align	2
	.global	chex
	.type	chex, %function
chex:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	@ link register save eliminated.
	sub	r3, r0, #97
	cmp	r3, #25
	subls	r0, r0, #32
	andls	r0, r0, #255
	sub	r3, r0, #48
	cmp	r3, #9
	bhi	.L83
	and	r0, r0, #15
	bx	lr
.L83:
	sub	r3, r0, #65
	cmp	r3, #5
	subls	r0, r0, #55
	andls	r0, r0, #255
	mvnhi	r0, #0
	bx	lr
	.size	chex, .-chex
	.align	2
	.global	htol
	.type	htol, %function
htol:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r3, r4, r5, r6, r7, r8, sl, lr}
	mov	r8, r0
	mov	r7, r1
	ldr	r4, [r1, #0]
	mov	r6, #0
.L88:
	ldrb	r5, [r4], #1	@ zero_extendqisi2
	mov	sl, r4
	mov	r0, r5
	bl	chex
	mov	r0, r0, asl #16
	mov	r2, r0, lsr #16
	cmn	r0, #65536
	movne	r2, r2, asl #16
	movne	r2, r2, asr #16
	orrne	r6, r2, r6, asl #4
	bne	.L88
.L90:
	str	r6, [r8, #0]
	str	r4, [r7, #0]
	mov	r0, r5
	ldmfd	sp!, {r3, r4, r5, r6, r7, r8, sl, lr}
	bx	lr
	.size	htol, .-htol
	.align	2
	.global	argck
	.type	argck, %function
argck:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r3, r4, r5, r6, r7, lr}
	mov	r6, r0
	mov	r5, r1
	mov	r7, r2
	mov	r4, #0
.L94:
	mov	r0, r4, asl #16
	add	r4, r4, #1
	mov	r4, r4, asl #16
	mov	r4, r4, lsr #16
	add	r0, r5, r0, asr #14
	mov	r1, r6
	bl	htol
	tst	r0, #223
	bne	.L92
	mov	r0, r4, asl #16
	mov	r0, r0, asr #16
	cmp	r0, r7
	mvnlt	r0, #0
	b	.L93
.L92:
	cmp	r0, #44
	beq	.L94
	mvn	r0, #0
.L93:
	ldmfd	sp!, {r3, r4, r5, r6, r7, lr}
	bx	lr
	.size	argck, .-argck
	.align	2
	.global	hex
	.type	hex, %function
hex:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	@ link register save eliminated.
	and	r0, r0, #15
	cmp	r0, #9
	addgt	r0, r0, #7
	add	r0, r0, #48
	bx	lr
	.size	hex, .-hex
	.align	2
	.global	ckasci
	.type	ckasci, %function
ckasci:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r3, lr}
	and	r0, r0, #255
	sub	r3, r0, #32
	mov	r3, r3, asl #16
	cmp	r3, #6160384
	movhi	r0, #46
	bl	SendData
	ldmfd	sp!, {r3, lr}
	bx	lr
	.size	ckasci, .-ckasci
	.align	2
	.global	cstrlen
	.type	cstrlen, %function
cstrlen:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	@ link register save eliminated.
	ldrb	r3, [r0, #0]	@ zero_extendqisi2
	cmp	r3, r1
	beq	.L104
	mov	r3, #1
.L103:
	ldrb	r2, [r0, r3]	@ zero_extendqisi2
	cmp	r2, r1
	beq	.L102
	add	r3, r3, #1
	cmp	r3, #80
	bne	.L103
	b	.L102
.L104:
	mov	r3, #0
.L102:
	mov	r0, r3
	bx	lr
	.size	cstrlen, .-cstrlen
	.align	2
	.global	csprintf
	.type	csprintf, %function
csprintf:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 56
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r4, r5, r6, r7, r8, r9, sl, fp, lr}
	sub	sp, sp, #60
	mov	r6, r0
	mov	r7, r1
	str	r2, [sp, #20]
	str	r3, [sp, #24]
	mov	r0, r1
	mov	r1, #0
	bl	cstrlen
	cmp	r0, #0
	str	r0, [sp, #8]
	ble	.L135
	mov	r1, #0
	str	r1, [sp, #12]
	mov	r5, r1
	mov	r3, r1
	add	sl, sp, #43
	ldr	r9, .L153
	add	r2, sp, #46
	str	r2, [sp, #4]
.L133:
	ldrb	r2, [r7, r3]	@ zero_extendqisi2
	cmp	r2, #37
	strneb	r2, [r6], #1
	addne	r5, r5, #1
	bne	.L117
	ldr	r4, [sp, #24]
	ldr	ip, [sp, #20]
	ldr	r1, [sp, #12]
	cmp	r1, #0
	moveq	r4, ip
	add	r1, r1, #1
	str	r1, [sp, #12]
	add	r8, r3, #1
	ldrb	r2, [r7, r8]	@ zero_extendqisi2
	sub	r1, r2, #48
	cmp	r1, #9
	bhi	.L137
	cmp	r2, #48
	addeq	r8, r3, #2
	ldreqb	r2, [r7, r8]	@ zero_extendqisi2
	moveq	r3, #0
	streq	r3, [sp, #16]
	movne	ip, #1
	strne	ip, [sp, #16]
	sub	r3, r2, #49
	cmp	r3, #8
	bhi	.L139
	and	r1, r2, #15
	add	r3, r8, #1
	ldrb	r2, [r7, r3]	@ zero_extendqisi2
	sub	r0, r2, #48
	cmp	r0, #9
	movhi	r8, r3
	bhi	.L110
	add	r1, r1, r1, asl #2
	and	r2, r2, #15
	add	r1, r2, r1, asl #1
	add	r8, r8, #2
	ldrb	r2, [r7, r8]	@ zero_extendqisi2
	b	.L110
.L137:
	mov	r1, #1
	str	r1, [sp, #16]
	b	.L110
.L139:
	mov	r3, #1
	str	r3, [sp, #16]
	mov	r1, r3
	b	.L110
.L110:
	and	r3, r2, #223
	cmp	r3, #76
	addeq	r8, r8, #1
	ldreqb	r2, [r7, r8]	@ zero_extendqisi2
	moveq	r0, #4
	movne	r0, #2
	and	r3, r2, #223
	cmp	r3, #88
	bne	.L113
	cmp	r2, #120
	beq	.L114
	mov	ip, #0
	str	ip, [sp, #32]
	mov	r2, #120
	b	.L119
.L113:
	cmp	r3, #68
	beq	.L142
	cmp	r2, #99
	streqb	r4, [r6], #1
	addeq	r5, r5, #1
	moveq	r3, r8
	beq	.L117
.L116:
	cmp	r2, #115
	bne	.L119
	b	.L143
.L120:
	strb	r3, [r6], #1
	add	r0, r0, #1
	cmp	r0, #78
	ble	.L118
	b	.L146
.L143:
	mov	r0, r5
	rsb	r4, r5, r4
.L118:
	ldrb	r3, [r4, r0]	@ zero_extendqisi2
	cmp	r3, #0
	bne	.L120
	mov	r5, r0
	mov	r3, r8
	b	.L117
.L119:
	cmp	r0, #2
	moveq	r4, r4, asl #16
	moveq	r4, r4, lsr #16
	cmp	r1, #10
	movge	r1, #10
	str	r1, [sp, #28]
	mov	r1, #0
	strb	r1, [sp, #54]
	cmp	r2, #120
	addne	r2, sp, #53
	bne	.L126
	mov	r2, #32
	strb	r2, [sp, #44]
	strb	r2, [sp, #45]
	add	fp, sp, #54
	str	r8, [sp, #36]
	ldr	r8, [sp, #32]
.L124:
	mov	r0, r4
	bl	hex
	cmp	r8, #0
	cmpne	r0, #57
	orrgt	r0, r0, #32
	strb	r0, [fp, #-1]!
	mov	r4, r4, lsr #4
	ldr	r3, [sp, #4]
	cmp	fp, r3
	bne	.L124
	ldr	r8, [sp, #36]
	b	.L125
.L126:
	umull	ip, r3, r9, r4
	mov	r3, r3, lsr #3
	add	r1, r3, r3, asl #2
	sub	r4, r4, r1, asl #1
	orr	r4, r4, #48
	strb	r4, [r2], #-1
	mov	r4, r3
	cmp	r2, sl
	bne	.L126
	b	.L125
.L130:
	add	r2, sp, #44
	ldrb	r2, [r3, r2]	@ zero_extendqisi2
	cmp	r2, #48
	bne	.L127
	cmp	r0, #0
	addne	r2, sp, #44
	movne	ip, #32
	strneb	ip, [r3, r2]
	b	.L128
.L127:
	cmp	r2, #32
	bne	.L129
.L128:
	add	r3, r3, #1
	cmp	r3, #9
	bne	.L130
.L129:
	rsb	r0, r3, #10
	ldr	r1, [sp, #28]
	cmp	r0, r1
	movlt	r0, r1
	add	ip, r5, r0
	cmp	ip, #78
	bgt	.L144
	rsb	r0, r0, #10
	cmp	r0, #9
	bgt	.L131
	mov	r3, r0
	rsb	r1, r0, r6
.L132:
	add	r2, sp, #44
	ldrb	r2, [r3, r2]	@ zero_extendqisi2
	strb	r2, [r1, r3]
	add	r3, r3, #1
	cmp	r3, #10
	bne	.L132
	rsb	r6, r0, r6
	add	r6, r6, #10
	b	.L131
.L117:
	cmp	r5, #78
	bgt	.L145
.L134:
	add	r3, r3, #1
	ldr	r2, [sp, #8]
	cmp	r2, r3
	bgt	.L133
	mov	r0, r5
	b	.L146
.L135:
	mov	r0, #0
	b	.L146
.L144:
	mov	r0, r5
	b	.L146
.L145:
	mov	r0, r5
	b	.L146
.L114:
	mov	r3, #1
	str	r3, [sp, #32]
	b	.L119
.L142:
	mov	r2, #100
	b	.L119
.L125:
	mov	r3, #0
	ldr	r0, [sp, #16]
	b	.L130
.L131:
	mov	r5, ip
	mov	r3, r8
	b	.L134
.L146:
	add	sp, sp, #60
	ldmfd	sp!, {r4, r5, r6, r7, r8, r9, sl, fp, lr}
	bx	lr
.L154:
	.align	2
.L153:
	.word	-858993459
	.size	csprintf, .-csprintf
	.align	2
	.global	cprintf
	.type	cprintf, %function
cprintf:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 80
	@ frame_needed = 0, uses_anonymous_args = 0
	str	lr, [sp, #-4]!
	sub	sp, sp, #84
	mov	lr, r0
	mov	ip, r1
	mov	r3, r2
	mov	r0, sp
	mov	r1, lr
	mov	r2, ip
	bl	csprintf
	add	r3, sp, #80
	add	r0, r3, r0
	mov	r3, #0
	strb	r3, [r0, #-80]
	mov	r0, sp
	bl	putscon
	add	sp, sp, #84
	ldr	lr, [sp], #4
	bx	lr
	.size	cprintf, .-cprintf
	.align	2
	.global	srload
	.type	srload, %function
srload:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 16
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r4, r5, r6, r7, r8, r9, sl, fp, lr}
	sub	sp, sp, #20
	str	r0, [sp, #8]
	mov	r9, #0
	str	r9, [sp, #4]
	mov	fp, r9
.L178:
	bl	RecvData
	cmp	r0, #3
	beq	.L158
	cmp	r0, #83
	bne	.L178
	bl	RecvData
	mov	r3, r0
	cmp	r0, #3
	beq	.L158
	cmp	r0, #57
	beq	.L160
	cmp	r0, #56
	beq	.L160
	cmp	r0, #55
	beq	.L160
	cmp	r0, #48
	beq	.L178
	cmp	r0, #53
	beq	.L178
	sub	r3, r0, #49
	cmp	r3, #2
	bhi	.L162
	and	sl, r0, #3
	bl	RecvData
	cmp	r0, #3
	beq	.L158
	bl	chex
	mov	r8, r0
	bl	RecvData
	cmp	r0, #3
	beq	.L158
	bl	chex
	adds	r8, r0, r8, asl #4
	beq	.L170
	rsb	r5, r8, #255
	sub	r4, r8, #1
	mov	r6, fp
.L163:
	bl	RecvData
	cmp	r0, #3
	beq	.L158
	bl	chex
	mov	r7, r0
	bl	RecvData
	cmp	r0, #3
	beq	.L158
	mov	r7, r7, asl #4
	bl	chex
	add	r0, r7, r0
	rsb	r5, r0, r5
	add	r6, r0, r6, asl #8
	sub	r4, r4, #1
	add	r3, r4, #2
	add	r3, r3, sl
	cmp	r8, r3
	bne	.L163
	ldr	r2, [sp, #8]
	ldr	r3, [r2, #0]
	add	r6, r3, r6
	ldr	r3, [sp, #4]
	cmp	r3, #0
	streq	r6, [sp, #12]
	moveq	r2, #1
	streq	r2, [sp, #4]
	cmp	r4, #0
	beq	.L165
	mov	r7, r9
	rsb	r9, r9, r6
.L166:
	bl	RecvData
	cmp	r0, #3
	beq	.L158
	bl	chex
	mov	r8, r0
	bl	RecvData
	cmp	r0, #3
	beq	.L158
	mov	r8, r8, asl #4
	bl	chex
	add	r0, r8, r0
	rsb	r5, r0, r5
	strb	r0, [r9, r7]
	add	r7, r7, #1
	subs	r4, r4, #1
	bne	.L166
	mov	r9, r7
.L165:
	bl	RecvData
	cmp	r0, #3
	beq	.L158
	bl	chex
	mov	r4, r0
	bl	RecvData
	cmp	r0, #3
	beq	.L158
	mov	r4, r4, asl #4
	bl	chex
	add	r4, r4, r0
	eor	r3, r4, r5
	tst	r3, #255
	beq	.L178
	ldr	r0, .L179
	bl	putscon
	mov	r1, r5, asl #16
	mov	r2, r4, asl #16
	ldr	r0, .L179+4
	mov	r1, r1, asr #16
	mov	r2, r2, asr #16
	bl	cprintf
	ldr	r0, .L179+8
	bl	putscon
	b	.L156
.L162:
	ldr	r0, .L179+12
	bl	putscon
	b	.L156
.L173:
	bl	RecvData
	subs	r4, r4, #1
	bne	.L173
.L169:
	ldr	r0, .L179+16
	bl	putscon
	ldr	r0, .L179+20
	ldr	r1, [sp, #12]
	bl	cprintf
	ldr	r0, .L179+24
	bl	putscon
	ldr	r0, .L179+28
	mov	r1, r9
	bl	cprintf
	b	.L156
.L158:
	ldr	r0, .L179+32
	bl	putscon
	b	.L156
.L170:
	mov	r3, #57
.L160:
	and	r3, r3, #15
	rsb	r4, r3, #14
	mov	r4, r4, asl #1
	cmp	r4, #0
	bgt	.L173
	b	.L169
.L156:
	add	sp, sp, #20
	ldmfd	sp!, {r4, r5, r6, r7, r8, r9, sl, fp, lr}
	bx	lr
.L180:
	.align	2
.L179:
	.word	.LC2
	.word	.LC3
	.word	.LC4
	.word	.LC5
	.word	.LC6
	.word	.LC7
	.word	.LC8
	.word	.LC9
	.word	.LC10
	.size	srload, .-srload
	.align	2
	.global	verify
	.type	verify, %function
verify:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r3, r4, r5, r6, r7, r8, r9, sl, fp, lr}
	ldr	r6, [r0, #0]
	ldr	r5, [r0, #4]
	ldr	r4, [r0, #8]
	cmp	r4, #0
	beq	.L182
	sub	r6, r6, #1
	sub	r5, r5, #1
	ldr	sl, .L186
	ldr	fp, .L186+4
	ldr	r9, .L186+8
.L184:
	ldrb	r8, [r6, #1]!	@ zero_extendqisi2
	ldrb	r7, [r5, #1]!	@ zero_extendqisi2
	cmp	r8, r7
	beq	.L183
	bl	crlf
	mov	r0, sl
	mov	r1, r6
	bl	cprintf
	mov	r0, fp
	mov	r1, r8
	bl	cprintf
	mov	r0, sl
	mov	r1, r5
	bl	cprintf
	mov	r0, r9
	mov	r1, r7
	bl	cprintf
	bl	CheckRecvData
	cmp	r0, #0
	beq	.L183
	bl	RecvData
	cmp	r0, #3
	beq	.L182
	bl	RecvData
	cmp	r0, #3
	beq	.L182
.L183:
	subs	r4, r4, #1
	bne	.L184
.L182:
	mov	r0, #0
	ldmfd	sp!, {r3, r4, r5, r6, r7, r8, r9, sl, fp, lr}
	bx	lr
.L187:
	.align	2
.L186:
	.word	.LC11
	.word	.LC12
	.word	.LC13
	.size	verify, .-verify
	.align	2
	.global	setmemo
	.type	setmemo, %function
setmemo:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 24
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r4, r5, r6, r7, r8, r9, sl, fp, lr}
	sub	sp, sp, #28
	mov	r4, r1
	tst	r1, #4
	andne	r4, r1, #3
	movne	sl, #1
	moveq	sl, #0
	cmp	r4, #1
	ldreq	r7, [r0, #0]
	biceq	r7, r7, #1
	streq	r7, [r0, #0]
	beq	.L191
.L190:
	cmp	r4, #2
	ldreq	r6, [r0, #0]
	ldrne	r5, [r0, #0]
.L191:
	cmp	r2, #1
	ble	.L193
	ldr	r3, [r0, #16]
	ldr	ip, [r0, #4]
	b	.L227
.L200:
	cmp	r4, #1
	streqh	ip, [r7], #2	@ movhi
	beq	.L195
	cmp	r4, #2
	streq	ip, [r6], #4
	beq	.L195
	cmp	r4, #0
	streqb	ip, [r5], #1
	b	.L195
.L195:
	ldr	r1, [r0, #12]
	add	ip, ip, r1
	subs	r3, r3, #1
	bne	.L227
	ldr	r3, [r0, #16]
	ldr	r1, [r0, #20]
	add	r5, r5, r1
	add	r7, r7, r1, asl #1
	add	r6, r6, r1, asl #2
	ldr	ip, [r0, #4]
	ldr	r1, [r0, #24]
	add	ip, ip, r1
	str	ip, [r0, #4]
.L227:
	ldr	r1, [r0, #8]
	sub	r8, r1, #1
	str	r8, [r0, #8]
	cmp	r1, #0
	bne	.L200
.L193:
	cmp	r2, #1
	bne	.L202
	ldr	r9, .L229
	mov	fp, #32
	add	r8, sp, #8
.L228:
	bl	crlf
	cmp	r4, #1
	bne	.L203
	mov	r0, r9
	mov	r1, r7
	bl	cprintf
	mov	r0, #61
	bl	SendData
	cmp	sl, #0
	beq	.L204
	b	.L205
.L203:
	cmp	r4, #2
	bne	.L206
	mov	r0, r9
	mov	r1, r6
	bl	cprintf
	mov	r0, #61
	bl	SendData
	cmp	sl, #0
	beq	.L207
	b	.L205
.L206:
	mov	r0, r9
	mov	r1, r5
	bl	cprintf
	mov	r0, #61
	bl	SendData
	cmp	sl, #0
	beq	.L208
	b	.L205
.L204:
	ldr	r0, .L229+4
	ldrsh	r1, [r7, #0]
	bl	cprintf
	b	.L205
.L207:
	mov	r0, r9
	ldr	r1, [r6, #0]
	bl	cprintf
	b	.L205
.L208:
	ldr	r0, .L229+8
	ldrb	r1, [r5, #0]	@ zero_extendqisi2
	bl	cprintf
.L205:
	mov	r0, fp
	bl	SendData
	mov	r0, r8
	mov	r1, #15
	mov	r2, #1
	bl	getscon
	cmn	r0, #1
	beq	.L202
	cmp	r0, #15
	ble	.L209
	bic	r0, r0, #32
	cmp	r0, #82
	bne	.L210
	cmp	r4, #1
	subeq	r7, r7, #2
	beq	.L228
.L211:
	cmp	r4, #2
	subeq	r6, r6, #4
	subne	r5, r5, #1
	b	.L228
.L210:
	cmp	r0, #88
	bne	.L214
	cmp	sl, #1
	bne	.L214
	cmp	r4, #1
	bne	.L215
	ldr	r0, .L229+4
	ldrsh	r1, [r7, #0]
	bl	cprintf
	b	.L216
.L215:
	cmp	r4, #2
	moveq	r0, r9
	ldreq	r1, [r6, #0]
	ldrne	r0, .L229+8
	ldrneb	r1, [r5, #0]	@ zero_extendqisi2
	bl	cprintf
.L216:
	mov	r0, fp
	bl	SendData
	b	.L228
.L214:
	cmp	r4, #1
	addeq	r7, r7, #2
	beq	.L219
.L218:
	cmp	r4, #2
	addeq	r6, r6, #4
	beq	.L228
.L219:
	add	r5, r5, #1
	b	.L228
.L209:
	str	r8, [sp, #0]
	ldrb	r3, [sp, #8]	@ zero_extendqisi2
	cmp	r3, #46
	beq	.L202
	add	r0, sp, #4
	mov	r1, sp
	bl	htol
	ldr	r3, [sp, #4]
	cmp	r4, #1
	streqh	r3, [r7], #2	@ movhi
	beq	.L228
.L220:
	cmp	r4, #2
	streq	r3, [r6], #4
	strneb	r3, [r5], #1
	b	.L228
.L202:
	mov	r0, #0
	add	sp, sp, #28
	ldmfd	sp!, {r4, r5, r6, r7, r8, r9, sl, fp, lr}
	bx	lr
.L230:
	.align	2
.L229:
	.word	.LC7
	.word	.LC14
	.word	.LC13
	.size	setmemo, .-setmemo
	.align	2
	.global	dumpm
	.type	dumpm, %function
dumpm:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 16
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r4, r5, r6, r7, r8, r9, sl, fp, lr}
	sub	sp, sp, #20
	mov	r9, r0
	ldr	r8, [r0, #0]
	bic	r8, r8, #15
	str	r8, [r0, #0]
	ldr	r4, [r0, #4]
	cmp	r1, #16
	bne	.L232
	rsb	r4, r8, r4
	add	r3, r4, #3
	cmp	r4, #0
	movlt	r4, r3
	mov	r4, r4, asr #2
	ldr	r5, .L246
	b	.L233
.L235:
	ldr	r3, [r9, #0]
	ldr	r1, [r3], #4
	str	r3, [r9, #0]
	mov	r0, r5
	bl	cprintf
	sub	r4, r4, #1
	ldr	r2, [r9, #0]
	ldr	r3, [r9, #4]
	cmp	r2, r3
	bge	.L231
.L233:
	cmp	r4, #0
	bne	.L235
.L232:
	bl	dumpmt
	add	r7, sp, #14
	ldr	sl, .L246+4
	mov	fp, #42
.L245:
	ldr	r0, .L246+8
	mov	r1, r8
	bl	cprintf
	mov	r5, r8
	sub	r4, sp, #2
	mov	r6, r4
.L238:
	ldrh	r3, [r5], #2
	mov	r8, r5
	strh	r3, [r6, #2]!	@ movhi
	mov	r3, r3, asl #16
	mov	r1, r3, asr #8
	mov	r0, sl
	orr	r1, r1, r3, lsr #24
	bl	cprintf
	bl	CheckRecvData
	cmp	r0, #0
	beq	.L237
	bl	RecvData
	cmp	r0, #3
	beq	.L231
	bl	RecvData
	cmp	r0, #3
	beq	.L231
.L237:
	cmp	r6, r7
	bne	.L238
	mov	r6, r8
	mov	r0, fp
	bl	SendData
.L239:
	ldrh	r5, [r4, #2]!
	mov	r5, r5, asl #16
	mov	r0, r5, asr #16
	bl	ckasci
	mov	r0, r5, asr #24
	bl	ckasci
	cmp	r4, r7
	bne	.L239
	mov	r0, fp
	bl	SendData
	ldr	r3, [r9, #4]
	cmp	r3, r6
	bls	.L231
	tst	r6, #15
	bne	.L245
	bl	crlf
	tst	r6, #255
	bne	.L245
	bl	dumpmt
	b	.L245
.L231:
	add	sp, sp, #20
	ldmfd	sp!, {r4, r5, r6, r7, r8, r9, sl, fp, lr}
	bx	lr
.L247:
	.align	2
.L246:
	.word	.LC9
	.word	.LC16
	.word	.LC15
	.size	dumpm, .-dumpm
	.align	2
	.global	findp
	.type	findp, %function
findp:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r4, r5, r6, r7, r8, r9, sl, lr}
	ldr	r4, [r0, #0]
	ldr	r8, [r0, #4]
	ldr	r5, [r0, #8]
	ldr	r7, [r0, #12]
	cmp	r5, #0
	beq	.L248
	cmp	r5, #1
	andeq	r7, r7, #255
	beq	.L251
.L250:
	cmp	r5, #2
	moveq	r7, r7, asl #16
	moveq	r7, r7, lsr #16
	beq	.L251
.L252:
	cmp	r5, #3
	biceq	r7, r7, #-16777216
.L251:
	bl	crlf
	cmp	r4, r8
	bne	.L258
	b	.L248
.L259:
	mov	r1, r6
	mov	r3, r6
.L254:
	mov	r2, r3, asl #16
	ldrb	r2, [r4, r2, asr #16]	@ zero_extendqisi2
	orr	r1, r2, r1, asl #8
	add	r3, r3, #1
	mov	r2, r3, asl #16
	mov	r3, r2, lsr #16
	cmp	r5, r2, asr #16
	bgt	.L254
.L257:
	cmp	r1, r7
	bne	.L255
	mov	r0, r9
	mov	r1, r4
	bl	cprintf
	add	r3, sl, #1
	mov	r3, r3, asl #16
	mov	sl, r3, lsr #16
	cmp	r3, #458752
	ble	.L256
	bl	crlf
	mov	sl, r6
.L256:
	bl	CheckRecvData
	cmp	r0, #0
	beq	.L255
	bl	RecvData
	cmp	r0, #3
	beq	.L248
	bl	RecvData
	cmp	r0, #3
	beq	.L248
.L255:
	add	r4, r4, #1
	cmp	r8, r4
	bne	.L253
	b	.L248
.L258:
	mov	sl, #0
	mov	r6, sl
	ldr	r9, .L261
.L253:
	cmp	r5, #0
	movle	r1, r6
	ble	.L257
	b	.L259
.L248:
	ldmfd	sp!, {r4, r5, r6, r7, r8, r9, sl, lr}
	bx	lr
.L262:
	.align	2
.L261:
	.word	.LC15
	.size	findp, .-findp
	.align	2
	.global	sievemain
	.type	sievemain, %function
sievemain:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r3, r4, r5, r6, r7, r8, r9, sl, fp, lr}
	ldr	r0, .L275
	bl	cprintf
	ldr	r0, .L275+4
	bl	cprintf
	bl	RecvData
	ldr	r3, .L275+8
	ldr	r2, [r3, #8]
	mov	r9, #1000
	ldr	fp, .L275+12
	mov	sl, #1
	ldr	r1, .L275+16
	mov	r4, #0
	ldr	lr, .L275+20
	ldr	r5, .L275+24
	b	.L264
.L265:
	strb	sl, [r3, #1]!
	cmp	r3, r1
	bne	.L265
	mov	r7, #3
	ldr	r6, .L275+28
	mov	ip, fp
	mov	r8, r4
.L269:
	ldrb	r3, [ip, #1]!	@ zero_extendqisi2
	cmp	r3, #0
	beq	.L266
	add	r3, r6, #-1073741824
	sub	r3, r3, #40960
	cmp	r3, lr
	bgt	.L267
	mov	r3, r6
.L268:
	strb	r4, [r3], r7
	add	r0, r5, r3
	cmp	r0, lr
	ble	.L268
.L267:
	add	r8, r8, #1
.L266:
	add	r6, r6, #3
	add	r7, r7, #2
	cmp	ip, r1
	bne	.L269
	subs	r9, r9, #1
	beq	.L270
.L264:
	mov	r3, fp
	b	.L265
.L270:
	ldr	r3, .L275+8
	ldr	r3, [r3, #8]
	ldr	r0, .L275+32
	mov	r1, r8
	rsb	r2, r2, r3
	bl	cprintf
	mov	r0, r8
	ldmfd	sp!, {r3, r4, r5, r6, r7, r8, r9, sl, fp, lr}
	bx	lr
.L276:
	.align	2
.L275:
	.word	.LC17
	.word	.LC18
	.word	-536854528
	.word	1073782783
	.word	1073790975
	.word	8191
	.word	-1073782784
	.word	1073782787
	.word	.LC19
	.size	sievemain, .-sievemain
	.align	2
	.global	main2
	.type	main2, %function
main2:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 8
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r4, r5, r6, r7, r8, r9, sl, fp, lr}
	sub	sp, sp, #12
	ldr	r3, .L312
	mov	r2, #17
	str	r2, [r3, #12]
	mov	r2, #1
	str	r2, [r3, #4]
	ldr	r0, .L312+4
	bl	putscon
	mov	r6, #0
	mov	r7, r6
	ldr	r5, .L312+8
	ldr	r4, .L312+12
	ldr	fp, .L312+16
	ldr	r9, .L312+20
	ldr	sl, .L312+24
.L311:
	mov	r0, r5
	bl	putscon
	mov	r0, r4
	mov	r1, #79
	mov	r2, #0
	bl	getscon
	cmn	r0, #1
	bne	.L279
	bl	crlf
	b	.L311
.L279:
	str	r4, [sp, #4]
	ldrb	r3, [r4, #0]	@ zero_extendqisi2
	sub	r3, r3, #83
	cmp	r3, #35
	ldrls	pc, [pc, r3, asl #2]
	b	.L281
.L292:
	.word	.L282
	.word	.L283
	.word	.L281
	.word	.L281
	.word	.L281
	.word	.L281
	.word	.L281
	.word	.L281
	.word	.L281
	.word	.L281
	.word	.L281
	.word	.L281
	.word	.L281
	.word	.L281
	.word	.L281
	.word	.L281
	.word	.L281
	.word	.L284
	.word	.L281
	.word	.L285
	.word	.L286
	.word	.L281
	.word	.L287
	.word	.L281
	.word	.L281
	.word	.L288
	.word	.L289
	.word	.L281
	.word	.L281
	.word	.L281
	.word	.L281
	.word	.L281
	.word	.L290
	.word	.L281
	.word	.L281
	.word	.L291
.L284:
	ldr	r3, .L312+28
	str	r3, [sp, #4]
	ldrb	r3, [r3, #0]	@ zero_extendqisi2
	cmp	r3, #120
	ldreq	r3, .L312+32
	streq	r3, [sp, #4]
	moveq	r8, #16
	movne	r8, #0
	ldr	r3, [sp, #4]
	ldrb	r3, [r3, #0]	@ zero_extendqisi2
	cmp	r3, #0
	bne	.L294
	ldr	r3, .L312+36
	ldr	r1, [r3, #4]
	ldr	r0, [r3, #0]
	rsb	r2, r0, r1
	add	r2, r2, #2
	bic	r2, r2, #1
	add	r0, r0, r2
	str	r0, [r3, #0]
	add	r2, r1, r2
	str	r2, [r3, #4]
	b	.L295
.L294:
	add	r0, sp, #4
	ldr	r1, .L312+36
	mov	r2, #1
	bl	argck
	cmn	r0, #1
	bne	.L296
	bl	paramer
	b	.L287
.L296:
	cmp	r0, #1
	ldreq	r3, .L312+36
	ldreq	r2, [r3, #0]
	addeq	r2, r2, #126
	streq	r2, [r3, #4]
.L295:
	ldr	r0, .L312+36
	mov	r1, r8
	bl	dumpm
	b	.L287
.L289:
	ldr	r3, .L312+28
	str	r3, [sp, #4]
	ldrb	r3, [r3, #0]	@ zero_extendqisi2
	and	r3, r3, #223
	cmp	r3, #87
	ldreq	r3, .L312+32
	streq	r3, [sp, #4]
	moveq	r8, #1
	beq	.L298
.L297:
	cmp	r3, #76
	ldreq	r3, .L312+32
	streq	r3, [sp, #4]
	moveq	r8, #2
	movne	r8, #0
.L298:
	add	r0, sp, #4
	ldr	r1, .L312+36
	mov	r2, #3
	bl	argck
	cmp	r0, #3
	beq	.L299
	bl	paramer
	b	.L287
.L299:
	ldr	r6, .L312
	ldr	r7, [r6, #8]
	ldr	r0, .L312+36
	mov	r1, r8
	bl	movem
	ldr	r6, [r6, #8]
	b	.L287
.L291:
	add	r0, sp, #8
	ldr	r3, .L312+28
	str	r3, [r0, #-4]!
	ldr	r1, .L312+36
	mov	r2, #3
	bl	argck
	cmp	r0, #3
	beq	.L300
	bl	paramer
	b	.L287
.L300:
	ldr	r6, .L312
	ldr	r7, [r6, #8]
	ldr	r0, .L312+36
	bl	verify
	ldr	r6, [r6, #8]
	b	.L287
.L285:
	add	r0, sp, #8
	ldr	r3, .L312+28
	str	r3, [r0, #-4]!
	ldr	r1, .L312+36
	mov	r2, #4
	bl	argck
	cmp	r0, #4
	beq	.L301
	bl	paramer
	b	.L287
.L301:
	ldr	r6, .L312
	ldr	r7, [r6, #8]
	ldr	r0, .L312+36
	bl	findp
	ldr	r6, [r6, #8]
	b	.L287
.L286:
	add	r0, sp, #8
	ldr	r3, .L312+28
	str	r3, [r0, #-4]!
	ldr	r1, .L312+36
	mov	r2, #1
	bl	argck
	cmn	r0, #1
	bne	.L302
	bl	paramer
	b	.L287
.L302:
	ldr	r6, .L312
	ldr	r7, [r6, #8]
	ldr	r3, .L312+36
	ldr	ip, [r3, #0]
	mov	lr, pc
	bx	ip
	mov	r1, r0
	ldr	r6, [r6, #8]
	ldr	r0, .L312+40
	bl	cprintf
	b	.L287
.L288:
	add	r0, sp, #8
	ldr	r3, .L312+28
	str	r3, [r0, #-4]!
	ldr	r1, .L312+36
	mov	r2, #1
	bl	argck
	cmn	r0, #1
	bne	.L303
	bl	paramer
	b	.L287
.L303:
	bl	crlf
	ldr	r0, .L312+36
	mov	r1, #1
	bl	srload
	b	.L287
.L290:
	ldr	r3, .L312+28
	str	r3, [sp, #4]
	ldrb	r3, [r3, #0]	@ zero_extendqisi2
	and	r3, r3, #223
	cmp	r3, #78
	ldreq	r3, .L312+32
	streq	r3, [sp, #4]
	moveq	r8, #4
	movne	r8, #0
	ldr	r3, [sp, #4]
	ldrb	r2, [r3, #0]	@ zero_extendqisi2
	and	r2, r2, #223
	cmp	r2, #87
	orreq	r8, r8, #1
	addeq	r3, r3, #1
	streq	r3, [sp, #4]
	beq	.L306
.L305:
	cmp	r2, #76
	orreq	r8, r8, #2
	addeq	r3, r3, #1
	streq	r3, [sp, #4]
.L306:
	ldr	r1, .L312+36
	mov	r2, #1
	str	r2, [r1, #8]
	mov	r3, #0
	str	r3, [r1, #12]
	str	r3, [r1, #16]
	str	r3, [r1, #20]
	str	r3, [r1, #24]
	add	r0, sp, #4
	bl	argck
	mov	r2, r0
	cmn	r0, #1
	bne	.L307
	bl	paramer
	b	.L287
.L307:
	ldr	r6, .L312
	ldr	r7, [r6, #8]
	ldr	r0, .L312+36
	mov	r1, r8
	bl	setmemo
	ldr	r6, [r6, #8]
	b	.L287
.L282:
	bl	sievemain
	mov	r1, r0
	ldr	r0, .L312+44
	bl	cprintf
	b	.L287
.L283:
	rsb	r2, r7, r6
	mov	r0, r9
	mov	r1, r2
	bl	cprintf
	mov	r0, sl
	bl	cprintf
	bl	crlf
	b	.L287
.L281:
	mov	r0, fp
	bl	cprintf
.L287:
	bl	crlf
	b	.L311
.L313:
	.align	2
.L312:
	.word	-536854528
	.word	.LC20
	.word	.LC21
	.word	combuf
	.word	.LC26
	.word	.LC24
	.word	.LC25
	.word	combuf+1
	.word	combuf+2
	.word	param
	.word	.LC22
	.word	.LC23
	.size	main2, .-main2
	.align	2
	.global	main
	.type	main, %function
main:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	stmfd	sp!, {r3, r4, r5, lr}
	bl	CPU_Initialize
	ldr	r4, .L315
	ldr	r3, [r4, #416]
	orr	r3, r3, #1
	str	r3, [r4, #416]
	ldr	r3, .L315+4
	mov	r5, #0
	str	r5, [r3, #4]
	str	r5, [r3, #68]
	ldr	r2, .L315+8
	mov	r1, #262144
	str	r1, [r2, #-4063]
	str	r5, [r2, #-4047]
	str	r1, [r2, #-4043]
	ldr	r2, [r3, #0]
	bic	r2, r2, #240
	orr	r2, r2, #80
	str	r2, [r3, #0]
	ldr	r3, [r4, #424]
	bic	r3, r3, #192
	str	r3, [r4, #424]
	ldr	r3, [r4, #196]
	orr	r3, r3, #8
	str	r3, [r4, #196]
	mvn	r3, #0
	ldr	r2, [r3, #-4083]
	bic	r2, r2, #64
	str	r2, [r3, #-4083]
	ldr	r2, [r3, #-4079]
	orr	r2, r2, #64
	str	r2, [r3, #-4079]
	ldr	r3, .L315+12
	mvn	r2, #127
	strb	r2, [r3, #12]
	mov	r2, #9
	strb	r2, [r3, #0]
	strb	r5, [r3, #4]
	mvn	r2, #62
	strb	r2, [r3, #40]
	mov	r2, #3
	strb	r2, [r3, #12]
	mov	r0, #79
	bl	SendData
	mov	r0, #75
	bl	SendData
	mov	r0, #10
	bl	SendData
	mov	r0, #13
	bl	SendData
	str	r5, [r4, #0]
	mov	r3, #4
	str	r3, [r4, #4]
	mov	r3, #2
	str	r3, [r4, #0]
	mov	r0, r5
	bl	main2
.L316:
	.align	2
.L315:
	.word	-534790144
	.word	-536690688
	.word	1073729535
	.word	-536821760
	.size	main, .-main
	.comm	param,32,4
	.comm	combuf,80,4
	.comm	combuf2f,4,4
	.section	.rodata.str1.4,"aMS",%progbits,1
	.align	2
.LC0:
	.ascii	"\015\012PARAMETER ERROR !\015\012\000"
	.space	2
.LC1:
	.ascii	"\015\012ADDR      0 1  2 3  4 5  6 7  8 9  A B  C D"
	.ascii	"  E F  ascii\015\012\000"
.LC2:
	.ascii	"check sum error !!  \000"
	.space	3
.LC3:
	.ascii	"%04X %04X\000"
	.space	2
.LC4:
	.ascii	"SUM CHECK ERROR\015\012\000"
	.space	2
.LC5:
	.ascii	"S1~S3 abarble\015\012\000"
.LC6:
	.ascii	"start loading address = \000"
	.space	3
.LC7:
	.ascii	"%08lX\000"
	.space	2
.LC8:
	.ascii	"\015\012END.     bytes loaded = \000"
	.space	1
.LC9:
	.ascii	"%08lX\015\012\000"
.LC10:
	.ascii	"????? S load stop\015\012\000"
.LC11:
	.ascii	"%08lX=\000"
	.space	1
.LC12:
	.ascii	"%02X \000"
	.space	2
.LC13:
	.ascii	"%02X\000"
	.space	3
.LC14:
	.ascii	"%04X\000"
	.space	3
.LC15:
	.ascii	"%08lX \000"
	.space	1
.LC16:
	.ascii	"%04X \000"
	.space	2
.LC17:
	.ascii	"\015\012Buffer = 0x4000A000 --> 0x4000BFFF\000"
	.space	3
.LC18:
	.ascii	"\015\012type return to do 1000 iterations:\000"
	.space	3
.LC19:
	.ascii	"\015\012%ld primes Timer0=%ld count\015\012\000"
.LC20:
	.ascii	"\015\012LPC2388 monitor 2011.12.5-1\015\012\000"
.LC21:
	.ascii	"LPC2388-Bug>\000"
	.space	3
.LC22:
	.ascii	"\015\012ret= %08lX\000"
	.space	3
.LC23:
	.ascii	"sieve=%ld \000"
	.space	1
.LC24:
	.ascii	"\015\012Timer HEX = %lX, %ld\000"
	.space	1
.LC25:
	.ascii	"\203\312s\000"
.LC26:
	.ascii	"default\000"
	.ident	"GCC: (GNU) 4.7.1"

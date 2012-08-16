program:begin _ _ _	program:end * N * N * N
program:bgnin 0 _ _	inc_1:crr * N * N * N
program:begin 1 _ _	inc_1:crr * N * N * N


inc_1:crr * 1 *	inc_1:crr * N 0 R * N
inc_1:crr * _ *	inc_1:rew * N 1 L * N
inc_1:crr * 0 *	inc_1:rew * N 1 L * N
inc_1:rew * 0 *	inc_1:rew * N 0 L * N
inc_1:rew * _ *	inc_1:end * N _ R * N

inc_1:end * * *	mov_1_2:cpy * N * N * N

mov_1_2:cpy * 0 _	mov_1_2:cpy * N 0 R 0 R
mov_1_2:cpy * 1 _	mov_1_2:cpy * N 1 R 1 R
mov_1_2:cpy * _ _	mov_1_2:rew * N _ L _ L
mov_1_2:rew * 0 0	mov_1_2:rew * N 0 L 0 L
mov_1_2:rew * 1 1	mov_1_2:rew * N 1 L 1 L
mov_1_2:rew * _ _	mov_1_2:end * N _ R _ R

mov_1_2:end * * * 	adv_0:crr * N * N * N

adv_0:crr * * 1		adv_0:chk * N * N 0 R
adv_0:chk * * 1 	adv_0:rew * N * N 1 L
adv_0:chk * * 0 	adv_0:rew * N * N 0 L
adv_0:chk * * _ 	adv_0:clr * N * N _ L
adv_0:clr * * 0 	adv_0:clr * N * N _ L
adv_0:clr * * 1 	adv_0:rew * N * N 1 L
adv_0:clr * * _ 	adv_0:rew * N * N _ N
adv_0:rew * * 0 	adv_0:rew * N * N 0 L
adv_0:rew * * 1 	adv_0:rew * N * N 1 L
adv_0:rew * * _		adv_0:end * N * N _ R
adv_0:crr * * 0		adv_0:crr * N * N 1 R

adv_0:end * * 0 	adv_0:crr * R * N * N
adv_0:end * * 1 	adv_0:crr * R * N * N
adv_0:end * * _	read:chk * N * N * N

read:chk 0 * * 	mem0:skp * R * N * N
read:chk 1 * * 	mem1:skp * R * N * N
read:chk # * * 	mem#:skp * R * N * N
mem0:skp 0 * * 	mem0:skp * R * N * N
mem0:skp 1 * * 	mem0:skp * R * N * N
mem0:skp # * * 	mov0_1_2:cpy * R * N * N
mem1:skp 0 * * 	mem1:skp * R * N * N
mem1:skp 1 * * 	mem1:skp * R * N * N
mem1:skp # * * 	mov1_1_2:cpy * R * N * N
mem#:skp 0 * * 	mov#_1_2:cpy * N * N * N
mem#:skp 1 * * 	mov#_1_2:cpy * N * N * N


mov0_1_2:cpy * 0 _	mov0_1_2:cpy * N 0 R 0 R
mov0_1_2:cpy * 1 _	mov0_1_2:cpy * N 1 R 1 R
mov0_1_2:cpy * _ _	mov0_1_2:rew * N _ L _ L
mov0_1_2:rew * 0 0	mov0_1_2:rew * N 0 L 0 L
mov0_1_2:rew * 1 1	mov0_1_2:rew * N 1 L 1 L
mov0_1_2:rew * _ _	mov0_1_2:end * N _ R _ R

mov0_1_2:end * * * 	adv0_0:crr * N * N * N

adv0_0:crr * * 1	adv0_0:chk * N * N 0 R
adv0_0:chk * * 1 	adv0_0:rew * N * N 1 L
adv0_0:chk * * 0 	adv0_0:rew * N * N 0 L
adv0_0:chk * * _ 	adv0_0:clr * N * N _ L
adv0_0:clr * * 0 	adv0_0:clr * N * N _ L
adv0_0:clr * * 1 	adv0_0:rew * N * N 1 L
adv0_0:clr * * _ 	adv0_0:rew * N * N _ N
adv0_0:rew * * 0 	adv0_0:rew * N * N 0 L
adv0_0:rew * * 1 	adv0_0:rew * N * N 1 L
adv0_0:rew * * _	adv0_0:end * N * N _ R
adv0_0:crr * * 0	adv0_0:crr * N * N 1 R

adv0_0:end * * _ 	comp0:chck * N * N * N
adv0_0:end * * 0 	adv0_0:crr * R * N * N
adv0_0:end * * 1 	adv0_0:crr * R * N * N
comp0:chck 0 * * 	comp0:succ * N * N * N
comp0:chck 1 * * 	comp0:fail * N * N * N

mov1_1_2:cpy * 0 _	mov1_1_2:cpy * N 0 R 0 R
mov1_1_2:cpy * 1 _	mov1_1_2:cpy * N 1 R 1 R
mov1_1_2:cpy * _ _	mov1_1_2:rew * N _ L _ L
mov1_1_2:rew * 0 0	mov1_1_2:rew * N 0 L 0 L
mov1_1_2:rew * 1 1	mov1_1_2:rew * N 1 L 1 L
mov1_1_2:rew * _ _	mov1_1_2:end * N _ R _ R

mov1_1_2:end * * * 	adv1_0:crr * N * N * N

adv1_0:crr * * 1	adv1_0:chk * N * N 0 R
adv1_0:chk * * 1 	adv1_0:rew * N * N 1 L
adv1_0:chk * * 0 	adv1_0:rew * N * N 0 L
adv1_0:chk * * _ 	adv1_0:clr * N * N _ L
adv1_0:clr * * 0 	adv1_0:clr * N * N _ L
adv1_0:clr * * 1 	adv1_0:rew * N * N 1 L
adv1_0:clr * * _ 	adv1_0:rew * N * N _ N
adv1_0:rew * * 0 	adv1_0:rew * N * N 0 L
adv1_0:rew * * 1 	adv1_0:rew * N * N 1 L
adv1_0:rew * * _	adv1_0:end * N * N _ R
adv1_0:crr * * 0	adv1_0:crr * N * N 1 R

adv1_0:end * * _ 	comp1:chck * N * N * N
adv1_0:end * * 0 	adv1_0:crr * R * N * N
adv1_0:end * * 1 	adv1_0:crr * R * N * N
comp1:chck 0 * * 	comp1:fail * N * N * N
comp1:chck 1 * * 	comp1:succ * N * N * N

mov#_1_2:cpy * 0 _	mov#_1_2:cpy * N 0 R 0 R
mov#_1_2:cpy * 1 _	mov#_1_2:cpy * N 1 R 1 R
mov#_1_2:cpy * _ _	mov#_1_2:rew * N _ L _ L
mov#_1_2:rew * 0 0	mov#_1_2:rew * N 0 L 0 L
mov#_1_2:rew * 1 1	mov#_1_2:rew * N 1 L 1 L
mov#_1_2:rew * _ _	mov#_1_2:end * N _ R _ R

mov#_1_2:end * * * 	adv#_0:crr * N * N * N

adv#_0:crr * * 1	adv#_0:chk * N * N 0 R
adv#_0:chk * * 1 	adv#_0:rew * N * N 1 L
adv#_0:chk * * 0 	adv#_0:rew * N * N 0 L
adv#_0:chk * * _ 	adv#_0:clr * N * N _ L
adv#_0:clr * * 0 	adv#_0:clr * N * N _ L
adv#_0:clr * * 1 	adv#_0:rew * N * N 1 L
adv#_0:clr * * _ 	adv#_0:rew * N * N _ N
adv#_0:rew * * 0 	adv#_0:rew * N * N 0 L
adv#_0:rew * * 1 	adv#_0:rew * N * N 1 L
adv#_0:rew * * _	adv#_0:end * N * N _ R
adv#_0:crr * * 0	adv#_0:crr * N * N 1 R

adv#_0:end * * _ 	comp#:chck * N * N * N
adv#_0:end * * 0 	adv#_0:crr * R * N * N
adv#_0:end * * 1 	adv#_0:crr * R * N * N
comp#:chck _ * * 	!program:acc * N * N * N

comp0:succ * * * 	rept:rew * L * N * N
comp1:succ * * * 	rept:rew * L * N * N
rept:rew 0 * * 		rept:rew * L * N * N
rept:rew 1 * * 		rept:rew * L * N * N
rept:rew # * * 		rept:rew * L * N * N
rept:rew _ * * 		inc_1:crr * R * N * N

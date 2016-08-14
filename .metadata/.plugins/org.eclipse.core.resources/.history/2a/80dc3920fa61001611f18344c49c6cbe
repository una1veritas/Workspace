copy	0 _ copy 	R 	0 R
copy	1 _ copy 	R 	1 R
copy 	# _ rew		N 	_ L

rew 	# 0	rew 	N 	0 L
rew 	# 1	rew 	N 	1 L
rew 	# _ add		R 	_ R

add		0 0 add 	R 	0 R
add 	0 1 add 	R 	1 R
add		0 _ add 	R 	0 R
add 	1 0 add 	R 	1 R
add 	1 1	addc	R 	0 R
add 	1 _ add 	R 	1 R
add		_ 0 add 	N 	0 R
add 	_ 1 add 	N 	1 R
add 	_ _ !stop	N 	_ N

addc 	0 0 add 	R 	1 R
addc 	0 _ add 	R 	1 R
addc 	_ 0 add 	N 	1 R
addc 	_ _ stop 	N 	1 R
addc 	0 1 addc	R 	0 R
addc 	_ 1 addc	N 	0 R
addc 	1 0 addc 	R 	0 R
addc 	1 _ addc 	R 	0 R
addc 	1 1 addc 	R 	1 R

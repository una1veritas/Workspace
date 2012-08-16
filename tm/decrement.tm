prg:ini	* *	cpy:bgn * N * N

cpy:bgn * * cpy:wrt * N * N
cpy:wrt 1 _	cpy:wrt 1 R 1 R
cpy:wrt 0 _	cpy:wrt 0 R 0 R
cpy:wrt _ _	cpy:rew _ N _ L
cpy:rew _ 0	cpy:rew _ N 0 L
cpy:rew _ 1	cpy:rew _ N 1 L
cpy:rew _ _	cpy:end _ N _ R

dec:bgn * *	dec:crr * N * N
dec:crr _ 1	dec:chk	_ N 0 R
dec:chk _ 1 dec:rew _ N 1 L
dec:chk _ 0 dec:rew _ N 0 L
dec:chk _ _ dec:clr _ N _ L
dec:clr _ 0 dec:clr _ N _ L
dec:clr _ 1 dec:rew _ N 1 L
dec:clr _ _ dec:rew _ N _ N
dec:rew _ 0 dec:rew _ N 0 L
dec:rew _ 1 dec:rew _ N 1 L
dec:rew _ _	dec:end _ N _ R
dec:crr _ 0	dec:crr _ N 1 R

dec:end * * ifzero * N * N

brn * _  !fin * N * N
brn * 0 	dec:bgn * N * N
brn * 1 	dec:bgn * N * N

inc:bgn * *	inc:crr * N * N
inc:crr * 0	inc:rew * N 1 L
inc:crr * 1 inc:crr * N 0 R
inc:crr * _ inc:rew * N 1 L
inc:rew * 0	inc:rew * N 0 L
inc:rew * 1	inc:rew * N 1 L
inc:rew * _ inc:end * N _ R
inc:end * * inc:crr * N * N

cpy:end * *	dec:bgn * N * N

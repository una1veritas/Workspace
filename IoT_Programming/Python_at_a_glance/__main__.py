import sys  # 繧ｳ繝｡繝ｳ繝医�ｯ # 縺ｮ蠕後↓譖ｸ縺擾ｼ景mport 縺ｯ繝ｩ繧､繝悶Λ繝ｪ縺ｮ隱ｭ縺ｿ霎ｼ縺ｿ謖�遉ｺ

def gcd(x, y):
    if x == 0 or y == 0 :
        return 0 # not defined
    while y != 1 :
        t = x % y
        x = y
        y = t
    return x

        
def sieve(n):                       # 髢｢謨ｰ縺ｨ蜷榊燕繧堤怐逡･縺吶ｋ蠑墓焚縺ｮ螳｣險��ｼ主ｼ墓焚縺ｫ蝙九�ｯ謖�螳壹＠縺ｪ縺�ｼ�
    primes = set(range(2, n+1) )    # 遽�蝗ｲ蝙九ｒ set 蝙九↓螟画鋤
    for i in range(2, int((n+1)/2) ) :
        mul = 2
        while (mul*i <= n+1) :
            if mul*i in primes :    # 隕∫ｴ�縺悟性縺ｾ繧後ｋ縺九←縺�縺九�ｮ蛻､螳夲ｼ弱�ｵ縺上∪繧後↑縺�蝣ｴ蜷井ｾ句､悶′逕溘§繧九�ｮ縺ｧ
                primes.remove(mul*i) 
            mul = mul + 1           # 繧､繝ｳ繧ｯ繝ｪ繝｡繝ｳ繝茨ｼ壹��縺昴ｓ縺ｪ繧ゅ�ｮ縺ｯ縺ｪ縺�
    return primes

print(gcd(41,93))
    
print("The number of command-line arguments:" + str(len(sys.argv))) 
                                    # len(sys.argv) 縺ｧ繝ｪ繧ｹ繝� sys.argv 縺ｮ髟ｷ縺輔ｒ蠕励ｋ�ｼ�
                                    # C 縺ｮ main 縺ｮ蠑墓焚 argc 縺ｫ逶ｸ蠖難ｼ�
                                    # 譁�蟄怜�励↓ +�ｼ育ｵ仙粋�ｼ峨〒縺阪ｋ縺ｮ縺ｯ譁�蟄怜�励↑縺ｮ縺ｧ str( ) 縺ｧ謨ｴ謨ｰ繧呈枚蟄怜�励↓螟画鋤
for i in range(0, 9) :              # for 螟画焚 in 遽�蝗ｲ�ｼ冗ｵ�ｼ上Μ繧ｹ繝茨ｼ城�榊�暦ｼ剰ｾ樊嶌 :�ｼ医さ繝ｳ繝橸ｼ� 縺ｧ螟画焚縺ｫ隕∫ｴ�縺碁��縺ｫ莉｣蜈･縺輔ｌ繧�
                                    # 縺薙�ｮ蝣ｴ蜷茨ｼ罫ange(0,9) 縺ｯ [0,1,...,7,8] 
    print(i, " ", end="")           # , 縺ｧ蛹ｺ蛻�繧九→繧ｹ繝壹�ｼ繧ｹ繧偵≠縺代※蜊ｰ蟄暦ｼ取隼陦後＆縺帙◆縺上↑縺�蝣ｴ蜷医�ｯ蜷榊燕莉倥″蠑墓焚 end="" 繧剃ｸ弱∴繧�
    if i < len(sys.argv) :          # if 譁�縺ｮ譚｡莉ｶ縺ｫ繧ｫ繝�繧ｳ縺ｯ荳崎ｦ�ｼ弱ヶ繝ｭ繝�繧ｯ縺ｯ : 縺ｮ谺｡縺ｮ陦後°繧蛾幕蟋�
        print(sys.argv[i])          # 縺輔ｉ縺ｫ繧､繝ｳ繝�繝ｳ繝医ｒ縺�繧後ｋ
    else:                           # else 繝悶Ο繝�繧ｯ繧ょ酔讒�
        print("(no argument)")
                                    # , 縺ｧ蛹ｺ蛻�繧九→隍�謨ｰ蠑墓焚縺ｨ縺励※髢薙ｒ縺ゅ￠縺ｦ蜊ｰ蟄�
print("finished.")                  # 譁�鬆ｭ縺ｮ繧､繝ｳ繝�繝ｳ繝医′ 0 縺ｫ繧ゅ←縺｣縺溘�ｮ縺ｧ for 繝ｫ繝ｼ繝励�ｮ螟�

if len(sys.argv) >= 1 :
    x = int(sys.argv[1])            # 螟画焚螳｣險�縺ｯ蠢�縺壹＠繧ょｿ�隕√↑縺�
    print(str(2) + u" 縺九ｉ " + str(x) + u" 縺ｾ縺ｧ縺ｮ髢薙�ｮ邏�謨ｰ")
                                    # 繝槭Ν繝√ヰ繧､繝域枚蟄励ｂ蜿ｯ�ｼ弱ョ繝輔か繝ｫ繝域枚蟄励さ繝ｼ繝峨′ unicode 縺ｧ縺ｪ縺�蝣ｴ蜷医�ｮ縺溘ａ u 繧偵▽縺代ｋ
    print(sieve(x))


import sys  # コメントは # の後に書く，import はライブラリの読み込み指示

def sieve(n):                       # 関数と名前を省略する引数の宣言．引数に型は指定しない．
    primes = set(range(2, n+1) )    # 範囲型を set 型に変換
    for i in range(2, int((n+1)/2) ) :
        mul = 2
        while (mul*i <= n+1) :
            if mul*i in primes :    # 要素が含まれるかどうかの判定．ふくまれない場合例外が生じるので
                primes.remove(mul*i) 
            mul = mul + 1           # インクリメント：　そんなものはない
    return primes
    
print("The number of command-line arguments:" + str(len(sys.argv))) 
                                    # len(sys.argv) でリスト sys.argv の長さを得る．
                                    # C の main の引数 argc に相当．
                                    # 文字列に +（結合）できるのは文字列なので str( ) で整数を文字列に変換
for i in range(0, 9) :              # for 変数 in 範囲／組／リスト／配列／辞書 :（コンマ） で変数に要素が順に代入される
                                    # この場合，range(0,9) は [0,1,...,7,8] 
    print(i, " ", end="")           # , で区切るとスペースをあけて印字．改行させたくない場合は名前付き引数 end="" を与える
    if i < len(sys.argv) :          # if 文の条件にカッコは不要．ブロックは : の次の行から開始
        print(sys.argv[i])          # さらにインデントをいれる
    else:                           # else ブロックも同様
        print("(no argument)")
                                    # , で区切ると複数引数として間をあけて印字
print("finished.")                  # 文頭のインデントが 0 にもどったので for ループの外

if len(sys.argv) >= 1 :
    x = int(sys.argv[1])            # 変数宣言は必ずしも必要ない
    print(str(2) + u" から " + str(x) + u" までの間の素数")
                                    # マルチバイト文字も可．デフォルト文字コードが unicode でない場合のため u をつける
    print(sieve(x))


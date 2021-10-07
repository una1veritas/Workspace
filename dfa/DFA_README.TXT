汎用有限オートマトンプログラム
dfa.c （Version 20211001.1）

1. コンパイル

dfa.c ファイルを GCC 等の C コンパイラで実行ファイルにコンパイルします。
例） Linux, Mac OS, Windows 10 Msys2/Mingw64 あるいは WSL 等で
gcc を使って実行ファイル dfa.exe ファイルにコンパイルする

% gcc -o dfa.exe dfa.c
% ls
dfa.c  dfa.exe
%

2. コマンドの実行

(1) 引数なしで実行すると、ヘルプの見方とともに、サンプルの定義（{a, b} 上の文字列で
文字 a の個数が偶数を受理）で動作する。
$ ./dfa.exe
define M by buily-in example: "0a1,0b0,1a0,1b1" "0" "0"
(Use 'command -h' to get a help message.)

dfa(
states = {'0', '1'},
alphabet = {'a', 'b'},
delta =
state symbol| next
------------+------
 '0' , 'a'  | '1'
 '0' , 'b'  | '0'
 '1' , 'a'  | '0'
 '1' , 'b'  | '1'
------------+------
initial state = '0'
accepting states = {'0'}
)
input 'abbaaabababb' :
   -> '0', -'a'-> '1', -'b'-> '1', -'b'-> '1', -'a'-> '0', -'a'-> '1', -'a'-> '0', -'b'-> '0', -'a'-> '1', -'b'-> '1', -'a'-> '0', -'b'-> '0', -'b'-> '0',
accepted.
Bye.

(2) -h をつけて実行すると、ヘルプ情報を表示する。
$ ./dfa.exe -h
usage: command "transition triples" "initial state" "final states" ("input string")
example: dfa.exe "0a1,0b0,1a0,1b1" "0" "0"

(3) 第１引数にコンマ区切りの遷移関数の定義列、第２引数に初期状態、第３引数に受理状態の列、
またオプションで第４引数に入力文字列を与えると、定義と遷移の列を表示し、結果を表示して終了する。
状態またはアルファベットに特殊な文字を使用しないならば、引数をダブルクオートでかこむ必要はない。
第４引数を省略すると、キーボードで改行を打つまでの入力を、行ごとに計算し結果を表示する。
空文字列を入力すると、プログラムを終了する。

遷移関数の定義は列は、
[遷移元の状態を表す文字][読む文字][遷移先の状態を表す文字]
をコンマで区切って並べた列である。
受理状態は、受理状態の集合に含まれる状態を表す文字を並べる。

$ ./dfa.exe 0a0,0b0,0c0,0h0,0.1,1.1,1a0,1b0,1c2,1h0,2a0,2b0,2c0,2h0,2.1 0 2
dfa(
states = {'0', '1', '2'},
alphabet = {'.', 'a', 'b', 'c', 'h'},
delta =
state symbol| next
------------+------
 '0' , '.'  | '1'
 '0' , 'a'  | '0'
 '0' , 'b'  | '0'
 '0' , 'c'  | '0'
 '0' , 'h'  | '0'
 '1' , '.'  | '1'
 '1' , 'a'  | '0'
 '1' , 'b'  | '0'
 '1' , 'c'  | '2'
 '1' , 'h'  | '0'
 '2' , '.'  | '1'
 '2' , 'a'  | '0'
 '2' , 'b'  | '0'
 '2' , 'c'  | '0'
 '2' , 'h'  | '0'
------------+------
initial state = '0'
accepting states = {'2'}
)
Type an input as a line, or quit by the empty line.
abca.c
input 'abca.c' :
   -> '0', -'a'-> '0', -'b'-> '0', -'c'-> '0', -'a'-> '0', -'.'-> '1', -'c'-> '2',
accepted.
abb.h
input 'abb.h' :
   -> '0', -'a'-> '0', -'b'-> '0', -'b'-> '0', -'.'-> '1', -'h'-> '0',
rejected.
abc.c.
input 'abc.c.' :
   -> '0', -'a'-> '0', -'b'-> '0', -'c'-> '0', -'.'-> '1', -'c'-> '2', -'.'-> '1',
rejected.

Bye.


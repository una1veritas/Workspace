# EMUZ80-6502RAM

![EMUZ80-6502RAM](https://github.com/satoshiokue/EMUZ80-6502RAM/blob/main/imgs/IMG_1725.jpeg)  
6502 Single-Board Computer    

![EMUZ80-6502RAM](https://github.com/satoshiokue/EMUZ80-6502RAM/blob/main/imgs/IMG_1711.jpeg)  
MEZ6502RAM and MEZ6502  

電脳伝説さん(@vintagechips)のEMUZ80が出力するZ80 CPU制御信号をメザニンボードで組み替え、W65C02Sと64kB RAMを動作させることができます。  
RAMの制御信号とメモリマップドIOのRDY信号をPICのCLC(Configurable Logic Cell)機能で作成しています。  
電源が入るとPICはW65C02Sを停止させ、ROMデータをRAMに転送します。データ転送完了後、バスの所有権をW65C02Sに渡してリセットを解除します。  

W65C02S6TPG-14とPIC18F47Q43の組み合わせで動作確認しています。  

動作確認で使用したCPU  
WDS W65C02S6TPG-14 1.6MHz - 10.6MHz  

```
NMOS6502にはBE信号がありません。起動時にPICからRAMにデータを転送できないためEMUZ80-6502RAMは使用できません。  
```

このソースコードは電脳伝説さんのmain.cを元に改変してGPLライセンスに基づいて公開するものです。

## メザニンボード
https://github.com/satoshiokue/MEZ6502RAM  

MEZ6502RAM専用プリント基板 - オレンジピコショップ  
https://store.shopping.yahoo.co.jp/orangepicoshop/pico-a064.html


## 回路図
https://github.com/satoshiokue/MEZ6502RAM/blob/main/MEZ6502RAM.pdf

## ファームウェア

EMUZ80で配布されているフォルダemuz80.X下のmain.cと置き換えて使用してください。
* emuz80_6502ram.c  

## クロック周波数

84行目のCLK_6502がクロック周波数です。初期値は2MHzになっています。
```
#define CLK_6502 2000000UL
```

## アドレスマップ
```
Memory
RAM1  0x0000 - 0xAFFF 44kバイト
RAM2  0xC000 - 0xFFFF 16kバイト

I/O   0xB000 - 0xBFFF
通信レジスタ 0xB019
制御レジスタ 0xB018
```

## PICプログラムの書き込み
EMUZ80技術資料8ページにしたがってPICに適合するファイルを書き込んでください。  

またはArduino UNOを用いてPICを書き込みます。  
https://github.com/satoshiokue/Arduino-PIC-Programmer

PIC18F47Q43 EMUZ80-6502RAM_Q43.hex  

PIC18F47Q83 EMUZ80-6502RAM_Q8x.hex  
PIC18F47Q84 EMUZ80-6502RAM_Q8x.hex  


```
6502 EhBASIC [C]old/[W]arm ?

Memory size ?

Enhanced BASIC 2.22p5a
44287 Bytes free

Ready
```
起動メッセージが出たら「c」キーを押します。Memory sizeでリターンキーを押すと利用可能な最大メモリでBASICの起動が完了します。


Enhanced 6502 BASIC by Lee Davison  
https://philpem.me.uk/leeedavison/6502/ehbasic/  

## W65C02Sプログラムの格納
インテルHEXデータを配列データ化して配列rom[]に格納すると0xC000に転送されW65C02Sで実行できます。

## 謝辞
思い入れのあるCPUを動かすことのできるシンプルで美しいEMUZ80を開発された電脳伝説さんに感謝いたします。

MEZ6502でやり残しを感じていたメモリ容量と動作速度の改善を達成することができました。  

## 参考）EMUZ80
EUMZ80はZ80CPUとPIC18F47Q43のDIP40ピンIC2つで構成されるシンプルなコンピュータです。

![EMUZ80](https://github.com/satoshiokue/EMUZ80-6502/blob/main/imgs/IMG_Z80.jpeg)

電脳伝説 - EMUZ80が完成  
https://vintagechips.wordpress.com/2022/03/05/emuz80_reference  
EMUZ80専用プリント基板 - オレンジピコショップ  
https://store.shopping.yahoo.co.jp/orangepicoshop/pico-a-051.html

## 改訂
Version 0.2 2023/4/19  
起動時にFirmwareの対象基板(MEZ6502RAM)とクロック周波数を表示  
IO処理を割り込みからポーリングに変更

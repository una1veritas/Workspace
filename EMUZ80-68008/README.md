# EMUZ80-68008

![MEZ68008](https://github.com/satoshiokue/EMUZ80-68008/blob/main/imgs/IMG_1537.jpeg)  
MC68008LC8  

![MEZ68008](https://github.com/satoshiokue/EMUZ80-68008/blob/main/imgs/IMG_1513.jpeg)  
MEZ68008  

![MEZ68008 Prototype](https://github.com/satoshiokue/EMUZ80-68008/blob/main/imgs/IMG_1490.jpeg)  
試作基板

電脳伝説さん(@vintagechips)のEMUZ80が出力するZ80 CPU信号をメザニンボードで組み替え、MC68008を動作させることができます。  
MC68008P10とPIC18F47Q83の組み合わせで動作確認しています。  

動作確認で使用したMPU  
MC68008P10  
MC68008LC8

ソースコードはGazelleさんのEMUZ80用main_cpz.cを元に改変してGPLライセンスに基づいて公開するものです。

https://drive.google.com/drive/folders/1NaIIpzsUY3lptekcrJjwyvWTBNHIjUf1

## メザニンボード
https://github.com/satoshiokue/MEZ68008

MEZ68008専用プリント基板 - オレンジピコショップ  
https://store.shopping.yahoo.co.jp/orangepicoshop/pico-a-061.html

## 回路図
https://github.com/satoshiokue/MEZ68008/blob/main/MEZ68008.pdf

## ファームウェア
EMUZ80で配布されているフォルダemuz80.X下のmain.cと置き換えて使用してください。  
ターゲットのPICを適切に変更してビルドしてください。  


## アドレスマップ
```
ROM   0x0000 - 0x3FFF 16Kbytes
RAM   0x8000 - 0x8FFF 4Kbytes (0x9FFF 8Kbytes:PIC18F47Q84,PIC18F47Q83)

UART  0xE000   Data REGISTER
      0xE001   Control REGISTER
```

## PICプログラムの書き込み
EMUZ80技術資料8ページにしたがってPICに適合するemuz80_68008_Qxx.hexファイルを書き込んでください。  

またはArduino UNOを用いてPICを書き込みます。  
https://github.com/satoshiokue/Arduino-PIC-Programmer

"Enhanced 68k BASIC Version 3.54"が起動します。

PIC書き込みがうまくいかない時はMEZ68008の/BRショートプラグを外してEMUZ80の電源を入れ直します。  

## 68008プログラムの改編
バイナリデータをテキストデータ化してファームウェアの配列rom[]に格納するとMC68008で実行できます。

テキスト変換例
```
xxd -i -c16 foo.bin > foo.txt
```

### EhBASIC68k for EMU
https://github.com/satoshiokue/EhBASIC68k-EMU

### TinyBASIC for EMU
https://github.com/satoshiokue/TinyBASIC-EMU

### unimon_EMUZ80-68008
https://github.com/satoshiokue/unimon_EMUZ80-68008

## 謝辞
思い入れのあるCPUを動かすことのできるシンプルで美しいEMUZ80を開発された電脳伝説さんに感謝いたします。

そしてEMUZ80の世界を発展させている開発者の皆さんから刺激を受けて68008に挑戦しています。

"POWER TO MAKE YOUR DREAM COME TRUE"

## 参考）EMUZ80
EUMZ80はZ80CPUとPIC18F47Q43のDIP40ピンIC2つで構成されるシンプルなコンピュータです。

![EMUZ80](https://github.com/satoshiokue/EMUZ80-6502/blob/main/imgs/IMG_Z80.jpeg)

電脳伝説 - EMUZ80が完成  
https://vintagechips.wordpress.com/2022/03/05/emuz80_reference  
EMUZ80専用プリント基板 - オレンジピコショップ  
https://store.shopping.yahoo.co.jp/orangepicoshop/pico-a-051.html

## 参考）phemu6809
comonekoさん(@comoneko)さんがEMUZ80にMC6809を搭載できるようにする変換基板とファームウェアphemu6809を発表されました。

![phemu6809](https://github.com/satoshiokue/EMUZ80-6502/blob/main/imgs/IMG_6809.jpeg)

https://github.com/comoneko-nyaa/phemu6809conversionPCB  
phemu6809専用プリント基板 - オレンジピコショップ  
https://store.shopping.yahoo.co.jp/orangepicoshop/pico-a-056.html


## 改訂
Version 0.2 2023/4/19  
起動時にFirmwareの対象基板(MEZ68008)とクロック周波数を表示  
10MHzファームウェアを追加  

# MEZ68K8_RAM<br>
<br>
EMUZ80で、MC68008を動かすメザニンボードとして、＠S_OkueさんのMEZ68008<br>
が2022年にGithubで公開されています。<br>
https://github.com/satoshiokue/MEZ68008
<br>
<br>
MEZ68008は、PIC18F47QXX（PIC18F47Q43/Q84/Q83）によってコントロール<br>
されており、メモリリソースもPIC18F47QXX内蔵のメモリを使用しています。<br>
とてもシンプルな構造となっており、68Kに初めて触れる人には最適と言えます。<br>
<br>
EMUZ80は、電脳伝説さんが開発し公開されているSBCです。Z80の制御にPIC18F57Q43を<br>
使用し、最小限度の部品構成でZ80を動かしています。<br>
<br>
＜電脳伝説 - EMUZ80が完成＞  <br>
https://vintagechips.wordpress.com/2022/03/05/emuz80_reference  <br>
<br>
このことがきっかけで、その後コアな愛好者によって、色々な拡張や<br>
新しいSBCが公開されています。<br>
<br>
今回、68008用に、512KBのメモリとSDカードI/Fを追加したメザニンボード、<br>
MEZ68K8_RAMを作成しました。MEZ68008と同様に、EMUZ80にアドオンすることで<br>
動作します。<br>
メモリとSDカードを追加することにより、CP/M-68Kを動かすことが出来ました。<br>
<br>

MEZ68K8_RAMを搭載したEMUZ80<br>
![MEZ68K8_RAM 1](photo/P1020614.JPG)
<br>

MEZ68K8_RAM拡大<br>
![MEZ68K8_RAM 2](photo/P1020610.JPG)

# 特徴<br>
<br>
・MPU : MC68008P10 10MHz<br>
・Microcontroller : PIC18F47Q43, PIC18F47Q84, PIC18F47Q83<br>
・512K SRAM搭載(AS6C4008-55PCN)<br>
・μSDカードI/F（SPI)<br>
・UART（9600bps無手順）<br>
・動作ソフト（起動時に選択可能）<br>
　　1) Enhanced 68k BASIC Version 3.54<br>
　　2) Universal Monitor 68000<br>
　　3) CP/M-68K<br>
<br>

Enhanced 68k BASICの起動画面<br>
![MEZ68K8_RAM 3](photo/basic68k.png)


ASCIIARTの実行結果<br>
![MEZ68K8_RAM 8](photo/ascii.png)


Universal Monitor 68000の起動画面<br>
![MEZ68K8_RAM 4](photo/unimon.png)


CP/M-68Kの起動画面<br>
![MEZ68K8_RAM 5](photo/cpm68k.png)


MEZ68K8_RAMシルク画像<br>
![MEZ68K8_RAM 6](photo/093906.png)


# ファームウェア（FW）
@hanyazouさんが作成したZ80で動作しているCP/M-80用のFWを<br>
(https://github.com/hanyazou/SuperMEZ80) 源流（ベース）にしています。<br>
今回は、MEZ88_RAM（https://github.com/akih-san/MEZ88_RAM） 用のFWを<br>
ベースにMEZ68K8_RAM用のFWとして動作するように修正を加えました。<br>
<br>
<br>
# CBIOSの開発環境
CP/M-68K Ver1.3は、[ここで](http://www.cpm.z80.de/binary.html)入手できます。<br>
また、[ここでも](http://www.easyaudiokit.com/bekkan2020/try_cpm68k/cpm68k.html)入手出来ます。<br>
CBIOS.BINは、DISK9にあるXBIOS.Cをベースに作成しました。<br>
Ｃコンパイラ、アセンブラは、Microtech Researchのクロス開発環境を使用しました。<br>
Internet Archiveウェブサイトから入手できます。<br>
https://archive.org/details/mri-68-k-c-cross-compiler-toolchain<br>
<br>
MSDOS上で動きますが、Windowsでは動作しないため、MSDOSが動く環境が必要になります。<br>
MEZ68K8_RAMでは、Windows上で動くエミュレーターとして有名なtakeda氏の<br>
msdos playerを使用しました。<br>
ここから（http://takeda-toshiya.my.coocan.jp/msdos/） 入手できます。<br>
こちらも、参考になるかと。<br>
http://iamdoingse.livedoor.blog/archives/24144518.html<br>
<br>
<br>
# CBIOS.BIN、CPM68K.BINを作成するために必要なツール<br>
<br>
・sed<br>
GNUの環境が必要になります。sedだけなら、ここから入手できます。<br>
https://sourceforge.net/projects/gnuwin32/files/sed/4.2.1/?sort=filename&sortdir=asc<br>
<br>
・bin2mot.exe、mot2bin.exe<br>
モトローラフォーマットのヘキサファイルとバイナリファイル相互変換ツール<br>
ソースとバイナリファイルは、ここから入手できます。<br>
https://sourceforge.net/projects/bin2mot/files/<br>
<br>

# その他のツール
・FWのソースのコンパイルは、マイクロチップ社の<br>
<br>
　「MPLAB® X Integrated Development Environment (IDE)」<br>
<br>
　を使っています。（MPLAB X IDE v6.20）コンパイラは、XC8を使用しています。<br>
(https://www.microchip.com/en-us/tools-resources/develop/mplab-x-ide)<br>
<br>
・universal moniter 68000、及びEnhanced 68k BASICは、Macro Assembler AS V1.42を<br>
　使用してバイナリを作成しています。<br>
　ここから(http://john.ccac.rwth-aachen.de:8000/as/) 入手できます。<br>
<br>
・FatFsはR0.15を使用しています。<br>
　＜FatFs - Generic FAT Filesystem Module＞<br>
　http://elm-chan.org/fsw/ff/00index_e.html<br>
<br>
・SDカード上のCP/Mイメージファイルの作成は、CpmtoolsGUIを利用しています。<br>
　＜CpmtoolsGUI - neko Java Home Page＞<br>
　http://star.gmobb.jp/koji/cgi/wiki.cgi?page=CpmtoolsGUI<br>
<br>

# 注意
WindowsのPowerShellを使用して68000のバイナリを作成しますが、その際にPowerShellの<br>
スクリプトファイル（拡張子.ps1）を使用しています。<br>
GitHubからソースファイルをダウンロードした際は、スクリプトファイルのマクロ実行禁止<br>
になっていますので、それを解除する必要があります。<br>
操作は、ファイルのプロパティを表示させて、セキュリティを許可します。<br>

スクリプトファイルの実行禁止解除<br>
![MEZ68K8_RAM 7](photo/propaty.png)

# 参考
＜EMUZ80＞<br>
EUMZ80はZ80CPUとPIC18F47Q43のDIP40ピンIC2つで構成されるシンプルなコンピュータです。<br>
（電脳伝説 - EMUZ80が完成）  <br>
https://vintagechips.wordpress.com/2022/03/05/emuz80_reference  <br>
<br>
＜SuperMEZ80＞<br>
SuperMEZ80は、EMUZ80にSRAMを追加しZ80をノーウェイトで動かすことができます。<br>
<br>
＜SuperMEZ80＞<br>
https://github.com/satoshiokue/SuperMEZ80<br>
<br>
＜＠hanyazouさんのソース＞<br>
https://github.com/hanyazou/SuperMEZ80/tree/mez80ram-cpm<br>
<br>
＜@electrelicさんのユニバーサルモニタ＞<br>
https://electrelic.com/electrelic/node/1317<br>

＜オレンジピコショップ＞  <br>
オレンジピコさんでEMUZ80、その他メザニンボードの購入できます。<br>
<br>
https://store.shopping.yahoo.co.jp/orangepicoshop/pico-a-051.html<br>
https://store.shopping.yahoo.co.jp/orangepicoshop/pico-a-061.html<br>
https://store.shopping.yahoo.co.jp/orangepicoshop/pico-a-062.html<br>
https://store.shopping.yahoo.co.jp/orangepicoshop/pico-a-079.html<br>
https://store.shopping.yahoo.co.jp/orangepicoshop/pico-a-087.html<br>

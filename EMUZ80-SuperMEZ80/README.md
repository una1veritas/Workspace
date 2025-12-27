# Firmware for SuperMEZ80-SPI/-CPM

EMUZ80用のメザニンボードSuperMEZ80用のファームウェアです。
SRAMとSD Card slotで、EMUZ80でCP/Mを動作させることができます。

## 特徴

* Satoshi Okueさん(@S_Okue)さんのSuperMEZ80にSD Card slotを追加してしています
* z80pack(https://github.com/udo-munk/z80pack) のSIMとある程度互換性のあるI/Oを実装しているため、
z80packのCP/Mのディスクイメージを起動することができます。  
（全てのI/Oが実装されているわけではありません）
* SuperMEZ80同様に、RAMの制御信号とIOアクセスのWAIT信号をPICのCLC(Configurable Logic Cell)機能で
作成しています。
* Z80に割り込み機能を使って、Z80動作中にPICからレジスタやメモリを読み出すことができます。
* 基板依存部分をsrc/boards以下に分離しており、EMUZ80と似た構成の基板への移植が容易です。

<img src="imgs/supermez80-cpm-and-emuz80.png" alt="SuperMEZ80-CPM and EMUZ80" width="100%">

Z80 Single-Board Computer EMUZ80 と SuperMEZ80-CPM

## 対応基板
複数のボードに対応しています。ビルド時の指定で切り替えます。

### MEZ80SD
Satoshi Okueさん(@S_Okue)さんのMEZ80RAMとEMUZ80-LEDを元にSPIインターフェースを追加し、SD Card slotを使えるようにしたもの。
オリジナルのMEZ80RAMと異なり、PICとSPIの/CSを接続する代わりにPICとZ80のA15が接続されていません。
このため、CP/Mを起動するには、DMA(Direct Memory Access)でなくPIO(Programmed I/O)のディスクアクセスを行う必要があります。  
https://github.com/satoshiokue/MEZ80SD  
ビルドパラメータ: BOARD=SUPERMEZ80_SPI PIC=18F47Q43

<img src="imgs/IMG_MEZ80SD.jpg" alt="MEZ80SD board" width="24%">

### SuperMEZ80-SPI
I/O expanderで、メモリのバンク切り替えや、NMIを使用することができます。
MEZ80SDではPICとSRAMのA14が接続されていますが、これを諦めて代わりにSPIの/CSに使用することにより、I/O expander (MCP23S08)を接続しています。
SPIのMOSI/MISO/SCKはZ80のデータバスと共用です。  
https://github.com/hanyazou/SuperMEZ80-SPI  
ビルドパラメータ: BOARD=SUPERMEZ80_SPI PIC=18F47Q43

<img src="imgs/IMG_SuperMEZ80-SPI.jpg" alt="SuperMEZ80-SPI board" width="24%">

### SuperMEZ80-CPM
I/O expanderなしで、メモリのバンク切り替えや、NMIを使用することができます。
部品点数が少なく、シンプルで組み立てやすい基板です。
構造が単純かつ、SD Card slotとZ80のデータバスを共有していないため、
SuperMEZ80-SPIより高いクロックで動作できます。
Z84C0020(20MHz版)との組み合わせで、PICのNCOで設定できる最高の16MHzでCP/M 3.0の動作実績があります。  
https://github.com/hanyazou/SuperMEZ80-CPM  
ビルドパラメータ: BOARD=SUPERMEZ80_CPM PIC=18F47Q43

<img src="imgs/IMG_SuperMEZ80-CPM.jpg" alt="SuperMEZ80-CPM board" width="24%">

### EMUZ80-57Q
@Gazelle8087 さんのEMUZ80強化版シングルボードコンピュータです。
オリジナルEMUZ80のPIC 18F47Q(40ピン)の代わりにPIC 18F57Q(48ピン)を使うことにより、
I/O expanderなしでSD Card slotのためのSPIと、Z80のアドレスバスA0~A15を接続しています。
残念ながらバンク切り替え機能がないので、SRAMは64KBまでです。  
https://twitter.com/Gazelle8087/status/1651571441727578112  
ビルドパラメータ: BOARD=EMUZ80_57Q PIC=18F57Q43

<img src="imgs/IMG_EMUZ80-57Q.jpg" alt="EMUZ80-57Q board" width="24%">

### Z8S180-57Q
@Gazelle8087 さんのEMUZ80-57Qを元にしたシングルボードコンピュータです。
ターゲットの CPU が Z80 から上位互換の Z8S180 に変更されています。
Z8S180にはZ80のクロック20MHzを大きく超える、33MHz版が存在するため、高速動作が期待できます。
Z8S180はMMUを内蔵していますが、本SuperMEZ80ではZ8S180のMMUをサポートしていないため、
残念ながらSRAMは64KBまでです。  
https://twitter.com/Gazelle8087/status/1708072678702379033  
ビルドパラメータ: BOARD=Z8S180_57Q PIC=18F57Q43

<img src="imgs/IMG_Z8S180-57Q.jpg" alt="Z8S180-57Q board" width="24%">

## ビルド

### ビルドの概要

* macOSかLinuxでビルドできます
* Windowsの場合、WSL2(Windows Subsystem for Linux)であればLinuxと同じ手順でビルドできます
* PIC18用のコンパイラ(XC8)を使います
* 以下のように make を実行すると、
PIC に書き込み可能なbuild/<基板名>.<PIC種別>/<基板名>-<PIC種別>.hexが作成されます
```
% git clone https://github.com/hanyazou/SuperMEZ80
% cd SuperMEZ80
% git submodule update --init --recursive
% make BOARD=SUPERMEZ80_CPM PIC=18F47Q43

% ls build/*.*/*.hex
-rw-r--r--  1 hanyazou  staff  218386 Aug 29 17:45 build/supermez80_cpm.18f47q43/supermez80_cpm-18f47q43.hex
```
* アセンブラを除く.cで書かれた部分だけであれば、ソースコードの取得、ビルド、書き込みを MPLAB X IDE で行うことができます

### Linux

ubuntu などの代表的なディストリビューションでビルドできます。
あらかじめ以下のようにして必要なパッケージをインストールしてください。

```
$ sudo apt update
$ sudo apt install build-essential curl xxd
```

MICROCHIPの
[Downloads Archive](https://www.microchip.com/en-us/tools-resources/archives/mplab-ecosystem)
からXC8コンパイラをインストールしてください。

開発および動作確認には、xc8 v2.36 を使用しています。
[xc8-v2.36-full-install-linux-x64-installer.run](https://ww1.microchip.com/downloads/aemDocuments/documents/DEV/ProductDocuments/SoftwareTools/xc8-v2.36-full-install-linux-x64-installer.run)
をダウンロードして、以下のように実行し、表示される手順に従ってインストールしてください。
インストールの詳細については
[Installing MPLAB® XC8 C Compiler](https://microchipdeveloper.com/xc8:installation)
を参照してください。

```
$ chmod +x xc8-v2.36-full-install-linux-x64-installer.run
$ sudo ./xc8-v2.36-full-install-linux-x64-installer.run
```

### Mac

XcodeやHome brewなどの一般的な開発環境に加えて、
MICROCHIPの
[Downloads Archive](https://www.microchip.com/en-us/tools-resources/archives/mplab-ecosystem)
からXC8コンパイラをインストールしてください。

開発および動作確認には、xc8 v2.40 を使用しています。
[xc8-v2.40-full-install-macos-x64-installer.dmg](https://ww1.microchip.com/downloads/aemDocuments/documents/DEV/ProductDocuments/SoftwareTools/xc8-v2.40-full-install-macos-x64-installer.dmg)
をダウンロードして、インストールしてください。
インストールの詳細については
[Installing MPLAB® XC8 C Compiler](https://microchipdeveloper.com/xc8:installation)
を参照してください。

### Windows

SuperMEZ80のビルドは、UNIX/LinuxスタイルのMakefileによって行われています。
Windowsに、WSL(Windows Subsystem for Linux)をインストールすることにより、
LinuxやMacと同じ手順でファームウェアをビルドすることができます。

<img src="imgs/windows+wsl.png" width="50%">

管理者権限で wsl --install を実行すると
デフォルトで WSL2 ubuntu がインストールされます。
追加の設定などは必要ありません。
[WSL を使用して Windows に Linux をインストールする方法](https://learn.microsoft.com/ja-jp/windows/wsl/install)
に従ってWSLをインストールしてください。

WSLでmakeを実行してファームウェアを作成すると、
\\\\wsl.localhost\\Ubuntu\\home\\hanyazou\\SuperMEZ80\\build...のあたりに.hexファイルができます。
("hanyazou"の部分は、WSLインストール時に指定したLinuxのユーザ名になります)
\\\\wsl.localhost\\Ubuntuを適当なネットワークドライブに割り当てておくと良いでしょう。

<img src="imgs/wsl-home-build-hex.png" width="50%">

### Docker

[Docker](https://www.docker.com/)を使用して、
ビルド時のホスト環境の違いによる影響を最小限にすることができます。

以下のようにdocker/build_env.shスクリプトを使用してmakeを実行するとubuntu 22.04のコンテナを
使ってビルドを実行します。

```
$ ./docker/build_env.sh make realclean test_build
```

初回実行時は、
ファームウェアのビルドの前にubuntuイメージの構築とXC8のインストールが行われます。
コマンドを実行した端末にXC8のライセンス条件などが表示されるのでよく読んで
Y/Nなどを入力してください。

### MPLAB X IDE
EMUZ80 などで使用する PIC 18F マイコンのMICROCHIP社が提供する統合開発環境です。  
Winodows, Mac, Linux に対応しており、
SuperMEZ80 のアセンブラソースコード（.z80, .asm）を除く、C言語で書かれた PIC の FW をビルドできます。  
（アセンブラを含む FW 全体の再ビルドには MPLAB X IDE 出なく、macOS や Linux の手順をお勧めします）

<img src="imgs/mplab-ide-0000.png" width="50%">

以下のMICROCHIPのサイトからMPLAB X IDE と XC8 コンパイラを
ダウンロードしてインストールしてください。
MPLAB X IDE v6.15 と MPLAB XC8 v2.41 の組み合わせでビルドできることを確認しています。  
https://www.microchip.com/en-us/tools-resources/develop/mplab-x-ide

MPLAB X IDE は、git クライアントの plug-in を備えており、MPLAB X IDE をインストールすれば、github から SuperMEZ80 のソースコードを clone して使用できます。
ソースコードの修正管理や、最新版への追従などを行う際には便利です。以下に、MPLAB X IDE を使って SuperMEZ80 のソースコードを clone する手順を示します。

Windows の MPLAB X IDE の Terminal で git コマンドを実行する場合は、cygwin をインストールしておいて下さい
https://www.cygwin.com/

<ol>
<li>MPLAB X IDE を起動します
<li>File > Close All Projects を選択して、すべてのプロジェクトを閉じます
<li>Team > Git > Clone ... を選択して、Clone Repository ダイアログを表示します<br>
  <img src="imgs/mplab-ide-0002.png" width="24%">
<li>Repositocy URL: に https://github.com/hanyazou/SuperMEZ80.git を指定します<br>
Clone into: には、ソースコードを配置する適当なフォルダを指定します<br>
  <img src="imgs/mplab-ide-0004.png" width="24%">
<li>Remote Branches は、mez80ram-cpm などを指定します（Select Allしても大丈夫です）<br>
  <img src="imgs/mplab-ide-0005.png" width="24%">
<li>Destination Directory は、ソースコードを配置する適当なフォルダを指定します<br>
Checkout Branch は、mez80ram-cpm を指定します<br>
Scan for NetBeans Projects after Clone にチェックが入っていることを確認してください<br>
  <img src="imgs/mplab-ide-0006.png" width="24%">
<li>Finish を押すと、github からソースコードを clone します<br>
途中、Do you want to automatically initialize and clone them? のようなダイアログが出るので、Yes を選択してください<br>
  <img src="imgs/mplab-ide-0008.png" width="24%">
<li>ソースコードの clone が終わったら、File > Open Project ... でダウンロードしたフォルダの mplab.X をオープンしてください<br>
  <img src="imgs/mplab-ide-0011.png" width="24%">
<li>左のペインの Projects で mplab を選んでから Window > IDE Tools > Terminal で cmd prompt を出して "git submodule update --init --recursive" を実行してください<br>
  <img src="imgs/mplab-ide-0017.png" width="24%">
<li>これでソースコードの準備は完了です。
ビルドおよびPICへの書き込み方法については、MPLAB X IDEの使い方を参照してください。
</ol>

## PICプログラムの書き込み
以下の書き込みツールのいずれかを使用して
ビルドしたファームウェアをEMUZ80上のPICマイコンに書き込みます。

* PICkit  
PICマイコンのMICROCHIP社純正ツールです。EMUZ80で使用するPIC 18Fに書き込むためには、PICkit 4以降の比較的新しいものが必要です。
（PICkit 2及び3は対応していません）  
MPLAB X IDEと連携できるので、MPLAB X IDE を使う場合には便利です。  
https://www.microchip.com/en-us/development-tool/PG164140
* PICkit minus  
PICkit 2及び3を使ってPIC 18Fに書き込むことができるソフトウェアです。  
http://kair.us/projects/pickitminus/
* Arduino-PIC-Programmer  
Arduino UNOを用いてPICを書き込みます。  
https://github.com/satoshiokue/Arduino-PIC-Programmer
* a-p-prog  
上記Arduino-PIC-Programmerの元になったツールです。
make upload を実行すると、こちらの a-p-prog を利用してPICにファームウェアを書き込みます。  
https://github.com/hanyazou/a-p-prog

## Z80プログラムの格納
SuperMEZ80ではインテルHEXデータを配列データ化して配列rom[]に格納すると0x0000に転送されZ80で実行できます。
SuperMEZ80-SPI用のファームウェアでは、rom[]に小さなプログラム(ipl.z80)が格納されいます。
これが実行されるとSDカードのディスクイメージの最初のセクタを読み込んで実行されます。

SDカードのディスクイメージは、
SDカードにCPMDISKSというフォルダを作成し、z80pack のディスクイメージか、または、
ファームウェアをビルドしてできたbuild/*/CPMDISKS.*/drivea.dskをコピーしておきます。
SDカードにCPMDISKSで始まる名前のフォルダが複数ある場合は、起動時に以下のように出力されるので、どれを使用するのか選択します。
```
Memory 000000 - 010000H 64 KB OK
0: CPMDISKS
1: CPMDISKS.3
2: CPMDISKS.PIO
3: CPMDISKS.180
M: Monitor prompt
Select[0]: 
```

通常とは異なるディスクイメージでないと起動しない場合は、デフォルトで適切なディスクイメージが選択される場合があります。

注意: SDカードはFAT32でフォーマットされている必要があります。
32GB以上の容量のSD XCで使用されるexFAT(FAT64)はサポートされません。
32GBより小さいサイズのSD HCカードを使用するか、またはPCなどでFAT32で再初期化したものを使用してください。

### ディスクイメージの修正について
ビルド時に作成される
build/*/CPMDISKS.*/drivea.dskは、z80packのCP/M 2.2用起動ディスクを修正したものです。
これらイメージの作成手順はMakefileを参照してください。

SuperMEZ80-SPI I/O expander付きなどを使用する場合は、
z80packのCP/M起動ディスクを無修正で使うことができます。
(プログラムI/Oも使用できます)

SuperMEZ80-SPI  I/O expander付きおよびSuperMEZ80-CPMでは、
AS6C4008などの2Mbit(256KB)以上のSRAMとの組み合わせで
banked biosのCP/M 3.0を起動することができます。

z80pack ディスクイメージの詳細は、udo-munk/z80packを参照してください。
https://github.com/udo-munk/z80pack

#### CPMDISKS.PIO/drivea.dsk
ディスクの読み書きをDMAでなく、プログラムI/Oに変更しています。
I/O expanderを使用できない場合は、CP/Mの起動にこのプログラムI/Oの修正が必要です。
具体的な修正内容は、cpm2 フォルダのboot.asm, bios.asmの履歴を参照してください。

#### CPMDISKS.180/drivea.dsk
Z8S180 は、CPU自体に幾つかの周辺機器や拡張レジスタを内蔵しており、これらのアクセスに 00h 付近の
I/O アドレスを使用します。
このためZ8S180-57Q基板は、z80packのディスクイメージとはメージとは異なるI/Oアドレスを使用する必要が
あります。
build/z8s180_57q.18f57q43/CPMDISKS.180/drivea.dsk は、シリアル入出力やSDカードへのアクセスを
通常の 00h 付近から 80h 付近に変更したものです。
具体的な修正内容は、cpm2 フォルダのboot\_z180.asm, bios\_z180.asmの履歴を参照してください。

## シリアルコンソール

シリアル端末は、cpm-v2.6.1 までは 9600 8N1、cpm-v2.7.0 以降のバージョンでは 115200 8N1 に設定してしださい。

## モニタ

端末から Break を入力するとモニタが起動し、
モニタのプロンプト "MON>" でモニタコマンドを実行できます。
モニタはターゲットCPU（Z80）ではなく、PICで実行されています。
help を入力すると使用できるコマンド一覧が表示されます。

## ファイル転送

シリアル端末から、CP/Mおよびモニタにファイルを転送することができます。

### CP/M から端末にファイルを転送する

CP/M とのファイル転送には CP/M のデバイスを使用します。
デバイスは z80pack の CP/M のものをそのまま使用します。
CP/M 2.2 では PUN:、CP/M 3.0 では AUX: を使用します。

デバイスから読み込むと、
以下のように YMODEM の転送開始を促す 'C' が繰り返し出力されます。
画面に 'C' が表示されたら、シリアル端末の  YMODEM ファイルアップロード機能で
ファイルを転送してください。

```
A>PIP TEST.TXT=AUX:
CCCCCCCCCCCCC
receiving file 'hello.txt', 15 bytes
total 1 file received


A>TYPE TEST.TXT
Hello, world!

A>
```

### 端末からCP/Mへファイルを転送する

CP/M 2.2 では RDR:、CP/M 3.0 では AUX: を使用します。
デバイスに書き込むと、
以下のように waiting for file transfer request via the terminal ... とメッセージが
表示されるので、シリアル端末の YMODEM ファイルダウンロード機能で
ファイルを転送してください。
ダウンロードされるファイル名は、PIP コマンドに指定したファイル名ではなく、
AUXOUT.TXT になります。

```
A>PIP AUX:=HELLO.TXT
waiting for file transfer request via the terminal ...
sending file 'AUXOUT.TXT' ...
total 1 file, 15 bytes sent

A>
```

### モニタでファイルを転送する

モニタでは、recv コマンドで端末からアップロード、send コマンドで端末にダウンロードできます。

### 制約
* この機能を利用するためには、
  シリアル端末がYMODEMプロトコルによるファイルのアップロード、
  ダウンロード機能を備えている必要があります。
* CP/Mの制約により、CP/Mとのファイルの送受信では、テキストファイルしか扱えません。
  00h は無視されます。
  また、CP/Mの CR,LF (ODh, OAh) は、端末側では LF (0Ah) に変換されます。

## サポート状況

| 基板           | RAM最大 | SD Card slot | CP/M 2.2 PIO | CP/M 2.2 DMA | CP/M 3.0 | モニタ | テスト     |
| ---            | ---     | ---          | ---          | ---          | ---      | ---    | ---        |
| SuperMEZ80-CPM | 256KB   | available    | ok           | ok           | ok       | ok     | 毎リリース |
| SuperMEZ80-SPI | 512KB   | available    | ok           | ok           | ok       | ok     | cpm-v2.7.0 |
| EMUZ80-57Q     | 64KB    | available    | ok           | ok           | n/a      | ok     | cpm-v2.7.0 |
| Z8S180-57Q     | 64KB    | available    | ok *2        | ok *1        | n/a      | NG *3  | cpm-v2.7.0 |
| MEZ80SD        | 64KB    | available    | ok           | n/a          | n/a      | n/a    | cpm-v2.7.0 |

*1 CPMDISKS.180/drivea.dsk が必要です  
*2 PIO 版の CPMDISKS.180/drivea.dsk を作成する必要があります（通常はDMAで使用してください）  
*3 安定して動作しません(2023-10-08)

## 謝辞
シンプルで美しいEMUZ80を開発された電脳伝説さんに感謝します。  
Z80Bを6MHzノーウェイトで動かしたSatoshi Okueさんに感謝します。  
またSPI接続もSatoshi OkueさんのMEZ80LEDを参考にしました。  

## ライセンス
元のソースコードは電脳伝説さんのmain.cを元に改変してGPLライセンスに基づいて公開されています。
新たに追加したソースコードは、扱いやすいようにMITライセンスです。
各ファイル冒頭のライセンスを参照してください。

## リファレンス
### EMUZ80
EUMZ80はZ80CPUとPIC18F47Q43のDIP40ピンIC2つで構成されるシンプルなコンピュータです。

<img src="imgs/IMG_Z80.jpeg" alt="EMUZ80" width="49%">

電脳伝説 - EMUZ80が完成  
https://vintagechips.wordpress.com/2022/03/05/emuz80_reference  
EMUZ80専用プリント基板 - オレンジピコショップ  
https://store.shopping.yahoo.co.jp/orangepicoshop/pico-a-051.html

### SuperMEZ80
SuperMEZ80は、EMUZ80にSRAMを追加し、Z80をノーウェイトで動かすことができるメザニンボードです

<img src="imgs/IMG_1595.jpeg" alt="SuperMEZ80" width="49%">

SuperMEZ80
https://github.com/satoshiokue/SuperMEZ80

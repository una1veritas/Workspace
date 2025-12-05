＜IOSYSの作成について＞<br>
<br>
8086のアセンブラは、Macro Assembler AS V1.42を使用しています。<br>
http://john.ccac.rwth-aachen.de:8000/as/<br>
<br>
「PowerShell」のスクリプトを使用するため、windowsのターミナルは、<br>
「Windows PowerShell」を使用してください。<br>
<br>
初期状態では、スクリプトの実行が許可されていないため、<br>
管理者権限でWindows PowerShellを起動し、以下の文を実行してください。<br>
<br>
PowerShell Set-ExecutionPolicy RemoteSigned<br>
<br>
上記コマンドを実行後は、管理者権限は必要ありません。<br>
ノーマル権限のWindows PowerShellでスクリプトが使えるようになります。<br>
<br>
後はノーマル権限でWindows PowerShellを立ち上げて、バッチコマンド<br>
mkiosys.batを実行すると、IO.SYSが作成されます。<br>
<br>
必要なファイル<br>
<br>
MEZ88IO.ASM<br>
mkiosys.bat<br>
mkiosys.ps1<br>
SYSINI_MES.bin<br>
<br>
＜その他のファイル＞<br>
<br>
MSDOS.SYS : MS-DOS Ver3.10 本体<br>

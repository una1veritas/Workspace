@echo off
Rem Do all settings depending from your environment here

rem Here you should set the type of your Pocket Computer
rem set SHARPC=1211
rem set SHARPC=1251
rem set SHARPC=1260
rem set SHARPC=1350
rem set SHARPC=1360
rem set SHARPC=1401
rem set SHARPC=1403
rem set SHARPC=1421
rem set SHARPC=1475
rem set SHARPC=1500
rem set SHARPC=1600
rem set SHARPC=E220
rem set SHARPC=E500
rem set SHARPC=G850

Rem For level inverting interfaces only
rem set SHARPDEV=INV

Rem set POCKTOOL path to the path\ of this script
set POCKTOOL=%~dp0
set BASDIR=%POCKTOOL%LISTINGS
set BINDIR=%BASDIR%
rem SET ORG=
set WAVDIR=%POCKTOOL%WAV
set TAPDIR=%WAVDIR%
Rem ANDDIR=?:\Android\data\o2s.emul.pc%SHARPC%\files\tapes

set EDITOR=notepad.exe
rem set EDITOR="%ProgramFiles(x86)%\TextPad 5\TextPad.exe"
rem set HEDITOR=C:\Win32app\HEXEdit.exe

echo Init %0.BAT
if "%SHARPC%"=="" goto Input
echo Pocket model: SHARP PC-%SHARPC%
rem pause
goto End

:Input
echo.
echo !!! Please edit %0.BAT to config your SHARP PC and environment !!!
set /p SHARPC=Pocket computer is SHARP PC-

:End

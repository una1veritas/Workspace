@rem This file is only usable for Ascii DATA, NOT for Basic sources
@echo Wav2Dasc is running...
@rem open console, change to directory
@rem and start batch with "wav2asc FILENAM" (without .wav)
@rem FileToOpen needed: http://www.horstmuc.de/win/wfile.zip
@rem Convert wave with SoX http://sox.sourceforge.net/
@echo off
if not exist GetFName.cmd (
	%~d0
	cd %~dp0
)
call SHARPSET
set fnamex=.WAV
set fnamedp=%WAVDIR%\
set fnamen=%1
set fnamef=%fnamedp%%fnamen%%fnamex%

Rem Note: Filenames with 1-7 characters are recommended!
if not exist %fnamef% call GetFName %fnamedp% "*%fnamex%;*.tap" "%1"
if not exist %fnamef% (
             echo Error: File %fnamef% does not exist! - Ctrl+C
             goto end
)

rem %WAVCNV% %fnamef% --norm=-1 %temp%\%fnamen%%fnamex%
rem --bits 8 
if errorlevel 1 goto end
rem pause
echo.

@echo on
wav2bin --pc=%SHARPC% --type=asc %fnamef% %BASDIR%\%fnamen%.ASC -l 0x40 %2 %3
@echo off
rem -l 0x40/0x400 ByteSum 0x80 TEXT 0x100 sync 0x200 BinEnd 0x8000 counter 1=Trans 2=SyncBits 4 ReadBits 8 SkipBits 0x10 Nibble 0x20 Byte 
if errorlevel 1 goto end


echo ---------------------------------------------
echo.
echo Please read the messages above!
pause
rem del %temp%\%fnamen%%fnamex%

start %EDITOR% %BASDIR%\%fnamen%.ASC
goto :EOF
:end
pause
@echo Wav2Tap is running...
@rem open console, change to directory
@rem and start batch with "wav2tap FILENAM" (without .wav)
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
set tapex=.tap

Rem Note: Filenames with 1-7 characters are recommended!
if not exist %fnamef% call GetFName %fnamedp% "*%fnamex%" "%1"
if not exist %fnamef% (
             echo Error: File %fnamef% does not exist! - Ctrl+C
             goto end
)
set sharpc2=%SHARPC:~0,2%
if "%sharpc2%" == "15" (
					   set tapex=_B.tap
Rem					   bin: _LM.tap, dat: _D.tap, bas: _B.tap
)
set sharpc2=

rem %WAVCNV% %fnamef% --norm=-1 %temp%\%fnamen%%fnamex%
rem --bits 8 
if errorlevel 1 goto end
rem pause
echo.

@echo on
wav2bin --pc=%SHARPC% --type=tap %fnamef% %TAPDIR%\%fnamen%%tapex% -l 0x40 %2 %3
@echo off
rem -l 0x40/0x400 ByteSum 0x80 TEXT 0x100 sync 0x200 BinEnd 0x8000 counter 1=Trans 2=SyncBits 4 ReadBits 8 SkipBits 0x10 Nibble 0x20 Byte 
if errorlevel 1 goto end


echo ---------------------------------------------
echo.
echo Please read the messages above!
pause
del %temp%\%fnamen%%fnamex%
copy %TAPDIR%\%fnamen%%tapex% %ANDDIR%\
Rem pause
echo Rename %TAPDIR%\%fnamen%%tapex% to CSAVENAME%tapex% 
echo Copy %TAPDIR%\%fnamen%%tapex% to your Android device into the app
echo directory %ANDDIR% and import it!
echo.
echo NOTE: 
echo 1. Import %fnamen%%tapex% from %ANDDIR%
echo    (%TAPDIR%),
echo    use the menu inside of your emulator application
echo 2. Start CLOAD "%fnamen%" inside the emulated BASIC Interpreter
echo.
pause
start %HEDITOR% %TAPDIR%\%fnamen%%tapex%
goto :EOF
:end
pause
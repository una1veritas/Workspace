@echo Wav2Img is running...
@rem open console, change to directory
@rem and start batch with "wav2img FILENAM" (without .wav)
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
rem copy %fnamef% %temp%\%fnamen%%fnamex%
rem %WAVCNV%  %fnamef% --norm=-1 %temp%\%fnamen%%fnamex%
rem %WAVCNV%  %fnamef% --norm=-1 --bits 8 %temp%\%fnamen%%fnamex%
if errorlevel 1 goto end
rem pause
echo.

rem @echo on
wav2bin --pc=%SHARPC% --type=img %fnamef% %BASDIR%\%fnamen%.IMG -l 0x40 %2 %3
@echo off
rem  -l 0x840 -l 0xC0  -l 0x2001 -l 0x0400   --type=rsv
if errorlevel 9 goto skip1
if errorlevel 1 goto end
:skip1
echo ---------------------------------------------
echo.
echo Please read the messages above!
pause
rem del %temp%\%fnamen%%fnamex%

start %HEDITOR% %BASDIR%\%fnamen%.IMG
goto :EOF
:end
pause

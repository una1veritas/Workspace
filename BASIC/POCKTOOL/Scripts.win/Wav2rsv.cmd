@echo Wav2Rsv is running...
@rem open console, change to directory
@rem and start batch with "wav2rsv FILENAM" (without .wav)
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

set fnamex2=.IMG
set sharpc2=%SHARPC:~0,2%
if "%sharpc2%" == "15" set fnamex2=.RSV
if "%sharpc2%" == "16" set fnamex2=.RSV
set sharpc2=

rem %WAVCNV%  %fnamef% --norm=-1 %temp%\%fnamen%%fnamex%
if errorlevel 1 goto error
rem pause
echo.

rem @echo on
wav2bin --type=rsv %fnamef% %BASDIR%\%fnamen%%fnamex2% --pc=%SHARPC% -l 0x40 %2 %3
@echo off
rem %temp%\%fnamen%%fnamex% -l 0x840 -l 0xC0  -l 0x2001 -l 0x0400   --type=rsv
rem if errorlevel 9 goto skip1
if errorlevel 1 goto end
:skip1
echo ---------------------------------------------
echo.
echo Please read the messages above!
pause
rem del %temp%\%fnamen%%fnamex%

start %HEDITOR% %BASDIR%\%fnamen%%fnamex2%
goto :EOF

:error
if errorlevel 10 goto E10
if errorlevel 9 goto E9
if errorlevel 8 goto E8
if errorlevel 7 goto E7
if errorlevel 6 goto E6
if errorlevel 5 goto E5
if errorlevel 4 goto E4
if errorlevel 3 goto E3
if errorlevel 2 goto E2
if errorlevel 1 goto E1
echo bad error value
goto end

:E10
echo To many minor errors found in the wave file!
echo Last error: %ERRORLEVEL%
goto end
:E9
echo Synchronisation not found or lost inside wave file!
goto end
:E8
echo Checksum error or unexpected end of file!
goto end
:E7
echo Format of the wave file or SHARP data is illegal!
goto end
:E6
echo Buffer/Memory overflow or to long lines!
goto end
:E5
echo File I/O-Error!
goto end
:E4
echo Invalid BASIC line numbers!
goto end
:E3
echo Invalid argument or unsupported device!
goto end
:E2
echo Invalid bit structure of a byte !
goto end
:E1
echo Argument Syntax Error!

:end
pause

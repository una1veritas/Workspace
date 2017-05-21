@echo Tap2Img is running...
@rem open console, change to directory
@rem and start batch with "tap2img FILENAM" (without .tap)
@rem FileToOpen needed: http://www.horstmuc.de/win/wfile.zip
@echo off
if not exist GetFName.cmd (
	%~d0
	cd %~dp0
)
call SHARPSET
set fnamex=.tap
set fnamedp=%TAPDIR%\
set fnamen=%1
set fnamef=%fnamedp%%fnamen%%fnamex%

Rem Note: Filenames with 1-7 characters are recommended!
if not exist %fnamef% call GetFName %fnamedp% "*%fnamex%" "%1"
if not exist %fnamef% (
             echo Error: File %fnamef% does not exist! - Ctrl+C
             goto end
)

rem echo Input: %fnamef%
rem @echo on
wav2bin --pc=%SHARPC% --type=img --tap %fnamef% %BASDIR%\%fnamen%.IMG -l 0x40 %2 %3
@echo off
rem  -l 0x840 -l 0xC0  -l 0x2001 -l 0x0400   --type=rsv
if errorlevel 9 goto skip1
if errorlevel 1 goto end
:skip1
echo ---------------------------------------------
echo.
echo Please read the messages above!
pause

start %HEDITOR% %BASDIR%\%fnamen%.IMG
goto :EOF
:end
pause

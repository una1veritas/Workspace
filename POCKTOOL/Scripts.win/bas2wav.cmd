@echo Bas2Wav is running...
@rem open console,
@rem change to directory and start batch with "bas2wav FILENAM"
@rem FileToOpen needed: http://www.horstmuc.de/win/wfile.zip

@echo off
if not exist GetFName.cmd (
	%~d0
	cd %~dp0
)
call SHARPSET
set fnamex=.BAS
set fnamedp=%BASDIR%\
set fnamen=%1
set fnamef=%fnamedp%%fnamen%%fnamex%

Rem Note: Filenames with 1-7 characters are recommended!
if not exist %fnamef% call GetFName %fnamedp% "*%fnamex%;*.PRG;*.ASC;*.SPP;*.SHA" "%1"
if not exist %fnamef% (
             echo Error: File %fnamef% does not exist! - Ctrl+C
             goto end
)

set fnamenS=%fnamen:~0,7%
set sharpc1=%SHARPC:~0,1%
set sharpc2=%SHARPC:~0,2%
if "%sharpc1%" == "E"  set fnamenS=%fnamen:~0,8%
if "%sharpc2%" == "G8" set fnamenS=%fnamen:~0,8%
if "%sharpc2%" == "15" set fnamenS=%fnamen:~0,16%
if "%sharpc2%" == "16" set fnamenS=%fnamen:~0,16%
if not "%fnamenS%" == "%fnamen%" (
             echo Note: Filename with less characters is recommended!
             rem goto end
)
set sharpc1=
set sharpc2=

bas2img --pc=%SHARPC% %fnamef% -l 0xC0
if errorlevel 1 goto end
rem  -l 0x80 0xA0
rem pause

echo.
@echo on
bin2wav --type=img --pc=%SHARPC% --name=%fnamenS% %fnamedp%%fnamen%.img %WAVDIR%\%fnamen%.wav -l 0x40 %2 %3
@echo off

if errorlevel 1 goto end

echo.
echo Start CLOAD and press a key ....
pause
del %fnamedp%%fnamen%.img
start %WAVDIR%\%fnamen%.wav
goto :EOF
:end
pause
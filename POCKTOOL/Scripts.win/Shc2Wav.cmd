@echo Shc2Wav is running...
@rem open console,
@rem change to directory and start batch with "shc2wav FILENAM"
@rem FileToOpen needed: http://www.horstmuc.de/win/wfile.zip

@echo off
if not exist GetFName.cmd (
	%~d0
	cd %~dp0
)
call SHARPSET
set fnamex=.SHC
set fnamedp=%BASDIR%\
set fnamen=%1
set fnamef=%fnamedp%%fnamen%%fnamex%

Rem Note: Filenames with 1-7 characters are recommended!
if not exist %fnamef% call GetFName %fnamedp% "*%fnamex%;*.IMG" "%1"
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
rem 
@echo on
bin2wav --type=shc --pc=%SHARPC% --name=%fnamenS% %fnamef% %WAVDIR%\%fnamen%.wav -l 0xC0 %2 %3
@echo off

if errorlevel 1 goto end

echo.
echo Start CLOAD (M) or INPUT# and press a key ....
pause
start %WAVDIR%\%fnamen%.wav
goto :EOF
:end
pause
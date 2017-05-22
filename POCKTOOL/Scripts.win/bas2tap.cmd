@echo Bas2Tap is running...
@rem open console,
@rem change to directory and start batch with "bas2tap FILENAM"
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
set tapex=.tap

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
if "%sharpc2%" == "16" set fnamenS=%fnamen:~0,8%
if "%sharpc2%" == "15" (
					   set fnamenS=%fnamen:~0,8%
					   set tapex=_B.tap
Rem					   bin: _LM.tap, dat: _D.tap, bas: _B.tap
)
if not "%fnamenS%" == "%fnamen%" (
             echo Note: Filename with less characters is recommended!
             rem goto end
)
set sharpc1=
set sharpc2=

bas2img --pc=%SHARPC% %fnamef%
if errorlevel 1 goto end
rem  -l 0x80 0xA0
rem pause

echo.
setlocal enabledelayedexpansion 
for %%a in ("a=A" "b=B" "c=C" "d=D" "e=E" "f=F" "g=G" "h=H" "i=I" "j=J" "k=K" "l=L" "m=M" "n=N" "o=O" "p=P" "q=Q" "r=R" "s=S" "t=T" "u=U" "v=V" "w=W" "x=X" "y=Y" "z=Z") do ( 
    set "fnamenS=!fnamenS:%%~a!" 
)
rem @echo on
bin2wav --type=img --pc=%SHARPC% --tap --name=%fnamenS% %fnamedp%%fnamen%.img %TAPDIR%\%fnamenS%%tapex% -l 0x40 %2 %3
@echo off
rem pause
if errorlevel 1 goto end

del %fnamedp%%fnamen%.img
copy %TAPDIR%\%fnamenS%%tapex% %ANDDIR%\
pause
echo Copy %TAPDIR%\%fnamenS%%tapex% to your Android device into the app
echo directory %ANDDIR% and import it!
endlocal
echo.
echo NOTE: 
echo 1. Import %fnamenS%%tapex% from %ANDDIR%
echo    (%TAPDIR%),
echo    use the menu inside of your emulator application
echo 2. Start CLOAD "%fnamenS%" inside the emulated BASIC Interpreter
pause
goto :EOF
:end
pause
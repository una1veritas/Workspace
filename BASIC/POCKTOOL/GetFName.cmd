@echo off
@rem get FileToOpen from: http://www.horstmuc.de/win/wfile.zip
@rem Params 1) Start path for FileToOpen 2) Extensions 
@rem     or 3) FullPathAndFileName from Drag and drop

if not %3 == "?" set fnamef=%3

if not exist %fnamef% (
	FileToOpen "set fnamef=" "%1%~2" "Select source file" > %temp%\fname.cmd 
	if errorlevel 1 goto :EOF
	call %temp%\fname.cmd
)
echo set fnamedp=%%~dp1>  %temp%\fname.cmd
echo set fnamen=%%~n1>>  %temp%\fname.cmd
echo set fnamex=%%~x1>>  %temp%\fname.cmd
call %temp%\fname.cmd %fnamef%
del %temp%\fname.cmd

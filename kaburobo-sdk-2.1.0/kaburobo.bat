@echo off

if "%KABUROBO_HOME%" == "" goto noEnv

if not exist "%KABUROBO_HOME%\lib\skaburobo-sdk.jar" goto noKaburoboSDKjar
if not exist "%KABUROBO_HOME%\lib\hsqldb.jar" goto noHsqljar

java -cp .;"%KABUROBO_HOME%\lib\hsqldb.jar";"%KABUROBO_HOME%\lib\skaburobo-sdk.jar";"%KABUROBO_HOME%\robot" jp.tradesc.superkaburobo.sdk.driver.RobotDriver %*

goto end

:noEnv
echo ŠÂ‹«•Ï” KABUROBO_HOME ‚ª’è‹`‚³‚ê‚Ä‚¢‚Ü‚¹‚ñB
goto end

:noKaburoboSDKjar
echo [%KABUROBO_HOME%\lib\skaburobo-sdk.jar] ‚ªŒ©‚Â‚©‚è‚Ü‚¹‚ñB
goto end

:noHsqljar
echo [%KABUROBO_HOME%\lib\hsqldb.jar] ‚ªŒ©‚Â‚©‚è‚Ü‚¹‚ñ
goto end

:end
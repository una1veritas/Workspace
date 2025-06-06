@echo off

if "%KABUROBO_HOME%" == "" goto noEnv

if not exist "%KABUROBO_HOME%\lib\skaburobo-sdk.jar" goto noKaburoboSDKjar
if not exist "%KABUROBO_HOME%\lib\hsqldb.jar" goto noHsqljar

java -cp .;"%KABUROBO_HOME%\lib\hsqldb.jar";"%KABUROBO_HOME%\lib\skaburobo-sdk.jar";"%KABUROBO_HOME%\robot" jp.tradesc.superkaburobo.sdk.driver.RobotDriver %*

goto end

:noEnv
echo 環境変数 KABUROBO_HOME が定義されていません。
goto end

:noKaburoboSDKjar
echo [%KABUROBO_HOME%\lib\skaburobo-sdk.jar] が見つかりません。
goto end

:noHsqljar
echo [%KABUROBO_HOME%\lib\hsqldb.jar] が見つかりません
goto end

:end
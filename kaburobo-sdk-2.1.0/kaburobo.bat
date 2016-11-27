@echo off

if "%KABUROBO_HOME%" == "" goto noEnv

if not exist "%KABUROBO_HOME%\lib\skaburobo-sdk.jar" goto noKaburoboSDKjar
if not exist "%KABUROBO_HOME%\lib\hsqldb.jar" goto noHsqljar

java -cp .;"%KABUROBO_HOME%\lib\hsqldb.jar";"%KABUROBO_HOME%\lib\skaburobo-sdk.jar";"%KABUROBO_HOME%\robot" jp.tradesc.superkaburobo.sdk.driver.RobotDriver %*

goto end

:noEnv
echo ���ϐ� KABUROBO_HOME ����`����Ă��܂���B
goto end

:noKaburoboSDKjar
echo [%KABUROBO_HOME%\lib\skaburobo-sdk.jar] ��������܂���B
goto end

:noHsqljar
echo [%KABUROBO_HOME%\lib\hsqldb.jar] ��������܂���
goto end

:end
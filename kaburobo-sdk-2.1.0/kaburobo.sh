#!/bin/bash
if [ -z "$KABUROBO_HOME" ]; then
  echo 環境変数 KABUROBO_HOME が定義されていません．;
elif [ ! -e "$KABUROBO_HOME/lib/skaburobo-sdk.jar" ]; then
  echo $KABUROBO_HOME/lib/skaburobo-sdk.jar が見つかりません．;
elif [ ! -e "$KABUROBO_HOME/lib/hsqldb.jar" ]; then
  echo $KABUROBO_HOME/lib/hsqldb.jar が見つかりません．;
else
  exec java -cp .:"$KABUROBO_HOME/lib/hsqldb.jar":"$KABUROBO_HOME/lib/skaburobo-sdk.jar":"$KABUROBO_HOME/robot" jp.tradesc.superkaburobo.sdk.driver.RobotDriver $*;
fi

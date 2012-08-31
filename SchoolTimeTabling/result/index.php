<?php
	session_start();
?>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=Shift_JIS">
<title>時間割の結果</title>
</head>
<body>
<?php
	// データベースに接続
	$con = mysql_connect("localhost", "yukiko", "nishimura")
		or die("<font color=$e_color>error : データベースの接続に失敗しました</font>");
		
	mysql_query("set names sjis") 
		or die("<font color=$e_color>error : キャラクタセットの変更に失敗しました</font><br>");
		
	// データベースを選択
	$select_db = mysql_select_db("schedule")
		or die("<font color=$e_color>error : データベースの選択に失敗しました</font>");
	
	// selectの実行
	//$sql = "SELECT project FROM status WHERE session='".session_id()."'";
	//$result = mysql_query ($sql) or die("SQL ".$sql." の実行に失敗しました<br>\n");
	// レコードの表示
	//if (mysql_numrows($result) != 1) {
	//	$currproject = "";
	//	print "プロジェクトが選択されていません<br>\n";
	//} else {
	//	$data = mysql_fetch_array($result);
	//	$currproject = $data['project'];
	//	print "現在使用中のプロジェクトは".$currproject . "です．<br>\n";
	//}
?>
<center>
<p><font color="#CCCCCC" size="4"><strong><font color="#FF3366">■</font><font color="#FF6666">■</font></strong></font><font size="4"><strong>　時間割案　</strong></font><font color="#CCCCCC" size="4"><strong><font color="#FF6666">■</font><font color="#FF3366">■</font></strong></font></p>
<?php
$e_color = "#FF0000";
$s_color = "#0000FF";

$fp_t = fopen("time.txt", "r");
if(!$fp_t){
	print("<font color=$e_color>error : ファイルを開くことができませんでした</font>");
	exit;
}
// ボタンが押されたときの時間
$a_time = fgets($fp_t, 8192); 
fclose($fp_t);

// 実行が終わったときの時間
$f_time = filemtime("../a/timetables/timetable0"); 

if($f_time < $a_time){
	print("<font color=$e_color>まだ、時間割ができていません。もうしばらくお待ちください。</font><br>\n");
	print("下の結果は前回実行したときの結果になります。<br><br>\n");
}

$fp = fopen("../a/config","r");
if(!$fp){
	print("<font color=$e_color>error : ファイルを開くことができませんでした</font>");
	exit;
}
$fs = filesize("../a/config");

print("表示したい時間割のボタンを押してください。<br><br>\n");

$i = 0;
$j = 0;
$buf = fgets($fp,$fs);

while($i < 3){
	if($buf[$j] == " "){
		$i++;
	}
	$j++;
	if($i == 3){
		$no = "";
		while($buf[$j] != " "){
			$no .= $buf[$j];
			$j++;
		}
	}
}

fclose($fp);

$table_color = "#FF6666";
$sel_color = "#FFFFFF";

print("<table border = \"0\" cellspacing=\"1\" cellpadding=\"3\" bgcolor=$table_color>\n");
print("<tr><td><div align=\"center\">時間割案</div></td></tr>\n");
for($l = 0; $l < $no; $l++){
	print("<tr bgcolor=$sel_color><td>第".($l+1)."候補　　</td>\n");
	print("<form name=\"\" method=\"post\" action=\"day_period.php\" target=\"_blank\"><td>\n");
	print("<input name=\"no\" type=\"hidden\" value=\"".($l+1)."\">\n");
	print("<input name=\"file\" type=\"hidden\" value=\"timetable".$l."\">\n");
	//print("<input name=\"currproject\" type=\"hidden\" value=\"".$currproject."\">\n");
	print("<input name=\"\" type=\"submit\" value=\"　表示　\">\n");
	print("</td></form></tr>\n");
}
print("</table><br>\n");
?>
<p>※出力ログは
    <a href="http://www.daisy.ai.kyutech.ac.jp/%7Eyukiko/a/timetables/result" target="_blank">
  こちら</a>  </p>
</center>
<a href="../index.php">←メニューに戻る</a></body>
</html>

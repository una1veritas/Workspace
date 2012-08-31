<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=Shift_JIS">
<title>講師×講義時間</title>
</head>

<body>
<?php 
$e_color = "#FF0000";
$s_color = "#0000FF";

print("<font color=$s_color>プロジェクト $currproject の第".$no."候補</font>の時間割です。[講師×講義時間]<br>\n");

// データベースに接続
$con = mysql_connect("localhost", "yukiko", "nishimura");
if(!$con){
	print("<font color=$e_color>error : データベースの接続に失敗しました</font>");
	exit;
	}

// データベースを選択
$select_db = mysql_select_db("schedule");
if(!$select_db){
	print("<font color=$e_color>error : データベースの選択に失敗しました</font>");
	exit;
	}

// 時間割を作成する
$f_name = "../a/timetables/".$file;
$fp = fopen($f_name, "r");
if(!$fp){
	print("<font color=$e_color>error : ファイルを開くことができませんでした</font>");
	exit;
}

$fs = filesize($f_name);
$i = 0;

// timetableをデータベースに登録
$dti = "delete from timetable";
$rti = mysql_query($dti);
$i = 0;
$buf = fgets($fp, $fs);
$iti = "insert into timetable (" . $buf . ") values (";
while($buf = fgets($fp, $fs)){
	$j = 0;
	$k = 0;
	$rti2 = mysql_query($iti . $buf . ")") or die("<font color=$e_color>error : 時間割のデータベースへの入力に失敗しました。</font>");
	while($j <= 3){
		if($buf[$k] != ","){
			$str[$i][$j] = $str[$i][$j] . $buf[$k];
		}
		else {
			$j++;
		}
		$k++;
	}
	$i++;
}
fclose($fp);


// 講義時間×講師テーブルの表示
$sell_color1 = "#6699FF";
$sell_color2 = "#CCCCFF";
$sell_color3 = "#CCCCCC";
print("<table width = \"1900\"border = \"1\" cellspacing=\"0\" cellpadding=\"0\"><BR>\n");
print("<tr><td width = \"150\" bgcolor=$sell_color3>　</td>\n");
$per = "select * from periods order by period_id";
$rper = mysql_query($per);
$nper = mysql_numrows ($rper);
$str = "\t";
for($i = 0; $i < $nper; $i++){
	$dper = mysql_fetch_array ($rper);
	if($i < 5){
		$sell_color1 = "#FFCCFF";
	}
	else if($i < 10){
		$sell_color1 = "#FFFF99";
	}
	else if($i < 15){
		$sell_color1 = "#CCFF99";
	}
	else if($i < 20){
		$sell_color1 = "#99CCFF";
	}
	else {
		$sell_color1 = "#9999FF";
	}
	print("<td bgcolor=$sell_color1 width = \"150\"><div align=\"center\">" . $dper["period_name"] . "</div></td>\n");
	$str .= $dper["period_name"];
	if($i == ($nper -1)){
		$str .= "\n";
	}
	else {
		$str .= "\t";
	}
}
print("</tr>\n");
$pro = "select * from processors where project = '".addslashes($currproject)."' order by processor_id";
$rpro = mysql_query($pro);
$npro = mysql_numrows ($rpro);
for($j = 0; $j < $npro; $j++){
	$dpro = mysql_fetch_array ($rpro);
	print("<tr>\n<td bgcolor=$sell_color2><div align=\"center\">". $dpro["processor_name"] . "</div></td>\n");
	$str .= $dpro["processor_name"] . "\t";
	for($k = 1; $k <= $nper; $k++){
		$select = "select task_id from timetable where period_id = " . $k . " and processor_id = " . $dpro["processor_id"];
		
		$rselect = mysql_query($select);
			$dsel = mysql_fetch_array ($rselect);
			if($dsel["task_id"] != ""){
				$task = "select * from tasks where project = '".addslashes($currproject)."' AND task_id = " . $dsel["task_id"];
				$rtask = mysql_query($task);
				$dtask = mysql_fetch_array ($rtask);
				$ntask = mysql_numrows ($rtask);
				print("<td><div align=\"center\">" .$dtask["task_title"] ."</div></td>\n");
				$str .= $dtask["task_title"];
			}
			else {
				print("<td><div align=\"center\">　　</div></td>\n");
			}
			if($k == $nper){
				$str .= "\n";
			}
			else {
				$str .= "\t";
			}	
	}
	print("</tr>\n");
}
print("</table>\n");

$file_name = "lecturer_period".$no.".txt";
$fp1 = fopen($file_name, "w");
if(!$fp1){
	print("<font color=$e_color>error : ファイルを開くことができませんでした</font>");
	exit;
}
fwrite($fp1, $str);

fclose($fp1);

// 切断する
mysql_close ($con);

$f_color = "#3366FF";
print("下のリンクを右クリックして、「対象をファイルに保存」で上の時間割のテキストファイルがダウンロードできます。\n");
print("<font color=$f_color><br>◆</font><a href=$file_name>この候補のテキストファイル</a>\n");
//print("<form name=\"\" method=\"post\" action=\"day_period.php\">\n");
//print("<input name=\"no\" type=\"hidden\" value=\"".$no."\">\n");
//print("<input name=\"file\" type=\"hidden\" value=\"".$file."\">\n");
//print("<input name=\"\" type=\"submit\" value=\" → 時間割表(時間×曜日) \">\n");
//print("</form>\n");

?>
</p>
</body>
</html>

<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=Shift_JIS">
<title>���ԁ~�j��</title>
</head>

<body>
<?php 
$e_color = "#FF0000";
$s_color = "#0000FF";

//print("<font color=$s_color>�v���W�F�N�g $currproject �� �� $no ���</font>�̎��Ԋ��ł��B[���ԁ~�j��]<br>\n");

// �f�[�^�x�[�X�ɐڑ�
$con = mysql_connect("localhost", "yukiko", "nishimura")
	or die("<font color=$e_color>error : �f�[�^�x�[�X�̐ڑ��Ɏ��s���܂���</font>");

	mysql_query("set names sjis") 
		or die("<font color=$e_color>error : �L�����N�^�Z�b�g�̕ύX�Ɏ��s���܂���</font><br>");
		
// �f�[�^�x�[�X��I��
$select_db = mysql_select_db("schedule")
	or die("<font color=$e_color>error : �f�[�^�x�[�X�̑I���Ɏ��s���܂���</font>");

// ���Ԋ����쐬����
$file = $_POST["file"];
$f_name = "../a/timetables/".$file;
$fp = fopen($f_name, "r");
if(!$fp){
	print("<font color=$e_color>error : �t�@�C�� $f_name ���J�����Ƃ��ł��܂���ł���</font>");
	exit;
}

$fs = filesize($f_name);
$i = 0;

// timetable���f�[�^�x�[�X�ɓo�^
$dti = "delete from timetable";
$rti = mysql_query($dti) or die("Failed to get results for SQL: " . $dti);
$i = 0;
$buf = fgets($fp, 8192);
$iti = "insert into timetable (" . $buf . ") values (";
while($buf = fgets($fp, 8192)){
	$j = 0;
	$k = 0;
	$rti2 = mysql_query($iti . $buf . ")") or die("Failed to get results for SQL: " . $iti . $buf . ")");
	while($j <= 3){
		if($buf[$k] != ",")
			$str[$i][$j] = $str[$i][$j] . $buf[$k];
		else 
			$j++;
		$k++;
	}
	$i++;
}
fclose($fp);

$period = "select * from periods";
$r_period = mysql_query($period)
	or die("<font color=$e_color>error : $period �f�[�^�x�[�X�̎��s�Ɏ��s���܂���</font>");

$p_num = mysql_num_rows($r_period);
unset($schedule);
unset($task_pro);
for($a = 1; $a <= $p_num; $a++){
	$search = "select * from timetable where period_id = $a order by task_id, processor_id";
	$result_s = mysql_query($search)
		or die("<font color=$e_color>error : $search �f�[�^�x�[�X�̎��s�Ɏ��s���܂���</font>");
	$b = 0;
	$c = 0;
	$d = 0;
	$task_n = -1;
	while($data = mysql_fetch_array($result_s)){
		if($task_n != $data["task_id"]){
			//$lecture = "select * from tasks where project = '".($currproject)."' AND task_id = ".$data["task_id"];
			$lecture = "select * from tasks where task_id = ".$data["task_id"];
			$r_lecture = mysql_query($lecture);
			if(!$r_lecture){
				print("<font color=$e_color>error : $lecture �f�[�^�x�[�X�̎��s�Ɏ��s���܂���</font>");
				exit;
			}
			$data2 = mysql_fetch_array($r_lecture);
			$schedule[$a][$b] = $data2["task_title"];
			$c = $b;
			$b++;
			$d = 0;
			$task_n = $data["task_id"];
		}
		//$lecturer = "select * from processors where project = '".($currproject)."' AND processor_id = ".$data["processor_id"];
		$lecturer = "select * from processors where processor_id = ".$data["processor_id"];
		$r_lecturer = mysql_query($lecturer);
		if(!$r_lecturer){
			print("<font color=$e_color>error : �f�[�^�x�[�X�̎��s�Ɏ��s���܂���</font>");
			exit;
		}
		$data3 = mysql_fetch_array($r_lecturer);
		$task_pro[$schedule[$a][$c]][$d] = $data3["processor_name"];
		$d++;
	}
}
	

// ���ԁ~�j���e�[�u���̕\��
$sell_color1 = "#6699FF";
$sell_color2 = "#CCCCFF";
$sell_color3 = "#CCCCCC";
$str = "\t";
print("<table border = \"1\" cellspacing=\"0\" cellpadding=\"0\"><BR>\n");
print("<tr><td bgcolor=$sell_color3>�@</td>\n");

$day = "select * from days order by day_id";
$rday = mysql_query($day);
$nday = mysql_numrows ($rday);
for($j = 0; $j < $nday; $j++){
	$dday = mysql_fetch_array ($rday);
	if($j == 0){
		$sell_color2 = "#FFCCFF";
	}
	else if($j == 1){
		$sell_color2 = "#FFFF99";
	}
	else if($j == 2){
		$sell_color2 = "#CCFF99";
	}
	else if($j == 3){
		$sell_color2 = "#99CCFF";
	}
	else {
		$sell_color2 = "#9999FF";
	}
	print("<td width = \"120\" bgcolor=$sell_color2><div align=\"center\">"
				. $dday["day_name"] . "</div></td>\n");
	if($j == ($nday - 1)){
		$str .= $dday["day_name"] . "\n";
	}
	else {
		$str .= $dday["day_name"] . "\t";
	}
}
print("</tr>\n");

for($i = 1; $i <= 5; $i++){
	print("<tr><td bgcolor=$sell_color2 width = \"30\"><div align=\"center\">".$i."</div></td>\n");
	$str .= $i . "\t";
	$arr = NULL;	// �e�L�X�g�t�@�C�����쐬���邽�߂Ƀf�[�^�ꎞ�ۑ��p
	$max = 0;		// ���Ԋ�1�s�̒��̍ő�s���𐔂���
	for($k = 0; $k < 5; $k++){
		$n = 0;			// 1�R�}�̒��̍s��
		print("<td><div align=\"center\">");
		$l = 0;
		while($schedule[($i+$k*5)][$l]){
			print("<br>\n<font color=$s_color><strong>" .$schedule[($i+$k*5)][$l]."</strong></font><br>\n");
			$arr[$k][$n] = $schedule[($i+$k*5)][$l];
			$n++;
			$m = 0;
			while($task_pro[$schedule[($i+$k*5)][$l]][$m]){
				print($task_pro[$schedule[($i+$k*5)][$l]][$m]."<br>\n");
				$arr[$k][$n] = $task_pro[$schedule[($i+$k*5)][$l]][$m];
				$n++;
				$m++;
			}	
			$l++;
		}
		print("�@</div></td>\n");
		if($n > $max){
			$max = $n;
		}
	}
	print("</tr>\n");
	
	if($max == 0){
		$str .= "\t\t\t\t\n";
	}
	for($p = 0; $p < $max; $p++){
		if($p != 0){
			$str .= "\t";
		}
		for($q = 0; $q < 5; $q++){
			$str .= $arr[$q][$p];
			if($q != 4){
				$str .= "\t";
			}
		}
		$str .= "\n";
	}
}
print("</table>\n");

$no = $_POST["no"];
$file_name = "day_period".$no.".txt";
$fp1 = fopen($file_name, "w");
if(!$fp1){
	print("<font color=$e_color>error : �t�@�C�� $file_name ���J�����Ƃ��ł��܂���ł���</font>");
	exit;
}
fwrite($fp1, $str);
fclose($fp1);

// �ؒf����
mysql_close ($con);


$f_color = "#3366FF";
print("���̃����N���E�N���b�N���āA�u�Ώۂ��t�@�C���ɕۑ��v�ŏ�̎��Ԋ��̃e�L�X�g�t�@�C�����_�E�����[�h�ł��܂��B\n");
print("<font color=$f_color><br>��</font><a href=$file_name>���̌��̃e�L�X�g�t�@�C��</a>\n");
print("<form name=\"\" method=\"post\" action=\"lecturer_period.php\">\n");
//print("<input name=\"currproject\" type=\"hidden\" value=\"".$currproject."\">\n");
print("<input name=\"no\" type=\"hidden\" value=\"".$no."\">\n");
print("<input name=\"file\" type=\"hidden\" value=\"".$file."\">\n");
print("<input name=\"\" type=\"submit\" value=\" �� ���Ԋ��\(�u�t�~�u�`����) \">\n");
print("</form>\n");

?>
</body>
</html>

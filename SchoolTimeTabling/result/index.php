<?php
	session_start();
?>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=Shift_JIS">
<title>���Ԋ��̌���</title>
</head>
<body>
<?php
	// �f�[�^�x�[�X�ɐڑ�
	$con = mysql_connect("localhost", "yukiko", "nishimura")
		or die("<font color=$e_color>error : �f�[�^�x�[�X�̐ڑ��Ɏ��s���܂���</font>");
		
	mysql_query("set names sjis") 
		or die("<font color=$e_color>error : �L�����N�^�Z�b�g�̕ύX�Ɏ��s���܂���</font><br>");
		
	// �f�[�^�x�[�X��I��
	$select_db = mysql_select_db("schedule")
		or die("<font color=$e_color>error : �f�[�^�x�[�X�̑I���Ɏ��s���܂���</font>");
	
	// select�̎��s
	//$sql = "SELECT project FROM status WHERE session='".session_id()."'";
	//$result = mysql_query ($sql) or die("SQL ".$sql." �̎��s�Ɏ��s���܂���<br>\n");
	// ���R�[�h�̕\��
	//if (mysql_numrows($result) != 1) {
	//	$currproject = "";
	//	print "�v���W�F�N�g���I������Ă��܂���<br>\n";
	//} else {
	//	$data = mysql_fetch_array($result);
	//	$currproject = $data['project'];
	//	print "���ݎg�p���̃v���W�F�N�g��".$currproject . "�ł��D<br>\n";
	//}
?>
<center>
<p><font color="#CCCCCC" size="4"><strong><font color="#FF3366">��</font><font color="#FF6666">��</font></strong></font><font size="4"><strong>�@���Ԋ��ā@</strong></font><font color="#CCCCCC" size="4"><strong><font color="#FF6666">��</font><font color="#FF3366">��</font></strong></font></p>
<?php
$e_color = "#FF0000";
$s_color = "#0000FF";

$fp_t = fopen("time.txt", "r");
if(!$fp_t){
	print("<font color=$e_color>error : �t�@�C�����J�����Ƃ��ł��܂���ł���</font>");
	exit;
}
// �{�^���������ꂽ�Ƃ��̎���
$a_time = fgets($fp_t, 8192); 
fclose($fp_t);

// ���s���I������Ƃ��̎���
$f_time = filemtime("../a/timetables/timetable0"); 

if($f_time < $a_time){
	print("<font color=$e_color>�܂��A���Ԋ����ł��Ă��܂���B�������΂炭���҂����������B</font><br>\n");
	print("���̌��ʂ͑O����s�����Ƃ��̌��ʂɂȂ�܂��B<br><br>\n");
}

$fp = fopen("../a/config","r");
if(!$fp){
	print("<font color=$e_color>error : �t�@�C�����J�����Ƃ��ł��܂���ł���</font>");
	exit;
}
$fs = filesize("../a/config");

print("�\�����������Ԋ��̃{�^���������Ă��������B<br><br>\n");

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
print("<tr><td><div align=\"center\">���Ԋ���</div></td></tr>\n");
for($l = 0; $l < $no; $l++){
	print("<tr bgcolor=$sel_color><td>��".($l+1)."���@�@</td>\n");
	print("<form name=\"\" method=\"post\" action=\"day_period.php\" target=\"_blank\"><td>\n");
	print("<input name=\"no\" type=\"hidden\" value=\"".($l+1)."\">\n");
	print("<input name=\"file\" type=\"hidden\" value=\"timetable".$l."\">\n");
	//print("<input name=\"currproject\" type=\"hidden\" value=\"".$currproject."\">\n");
	print("<input name=\"\" type=\"submit\" value=\"�@�\���@\">\n");
	print("</td></form></tr>\n");
}
print("</table><br>\n");
?>
<p>���o�̓��O��
    <a href="http://www.daisy.ai.kyutech.ac.jp/%7Eyukiko/a/timetables/result" target="_blank">
  ������</a>  </p>
</center>
<a href="../index.php">�����j���[�ɖ߂�</a></body>
</html>

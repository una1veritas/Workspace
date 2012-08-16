<?php
session_start();
//require_once('dbconnectvars.php');
//require_once('KyomuDB.php');
require_once('accessproc.php');

$error_msg = "";
// Check authenitication status and try a login.
if ( !isset($_COOKIE['user_id']) ) {
	if ( isset($_POST['submit']) ) {
		$kyomu = new KyomuAccessPage();
		if ( ($error_msg = $kyomu->open($_POST['username'], $_POST['password']) ) == "" ) {
			setcookie('user_id', $kyomu->user_id);
			setcookie('username', $kyomu->login_id);
			$GLOBALS["KYOMUACCESS"] = $kyomu;
			// Authenticated.
			header("Location: http://".$_SERVER['HTTP_HOST'].dirname($_SERVER['PHP_SELF'])."/index.php");
		} else {
			$GLOBALS["KYOMUACCESS"] = NULL;
			// Authentication failed.
			$error_msg .= "ユーザＩＤとパスワードが照合できません．" ;
		}
	}else {
		// Input incorrect
		$error_msg = "ユーザＩＤとパスワードが入力されていません．";
	}
} else {
   if ( isset($_POST['submit']) and isset($_POST['logout_yes']) ) {
      setcookie('user_id', "", time() -1);
      setcookie('username', "", time() -1);
	  $kyomu->close();
      header("Location: ". home_url("index.php") );
   }
   else 
   if ( empty($_COOKIE['user_id']) ) {
	 header("Location: ".home_url("index.php") );
   }
}
?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
<title>授業計画（シラバス）入力ページ</title>
<link href="./default.css" rel="stylesheet" type="text/css">
</head>
<div style="width: 640px;">

<?php
	//	echo "[".$_POST['username']."] [".$_POST['password']."]<BR/>";
//echo $_COOKIE['user_id'] . ", " . $_COOKIE['username']  . "<p/>";
if ( empty($_COOKIE['user_id']) ) {
	echo '<p class="error">'.$error_msg.'</p>';
	?>
	<p class="medium_serif">ログインしていません．</p>
	<p class="medsmall">ログインできないときは、ブラウザのクッキーが有効になっているか確認してください．</p>
	<form method="post"
		action="<?php echo "http://".$_SERVER['HTTP_HOST'].dirname($_SERVER['PHP_SELF'])."/index.php"; ?>">
		<fieldset>
			<legend>ログイン</legend>
			<label for="username">ユーザ名</label> <input type="text" id="username"
				name="username"
				value="<?php if (!empty($user_username)) echo $user_username; ?>" /><br />
			</label> <label for="password">パスワード</label> <input type="password"
				id="password" name="password" />
		</fieldset>
		<input type="submit" value="ログイン" name="submit" />
	</form>
<?php
   }
   else {
?>
<p class="login"><?php echo $_COOKIE['username']; ?>としてログインしています．</p>
<p><a href="./select.php">科目選択ページへ</a></p>
<?php
	//echo $kyomu->test();
?>
<p>
<form method="post" action="<?php echo "http://".$_SERVER['HTTP_HOST'].dirname($_SERVER['PHP_SELF'])."/index.php"; ?>">
<label class="medsmall">個人専用のＰＣでないときは，ログアウトするか，このページを見るのに使用しているウェブブラウザ（Internet Exproler や Safari など）を終了してください．<BR/>
<input type="submit" value="ログアウト" name="submit" />
</label>
<input type="hidden" value="logout_yes" name="logout_yes" />
</form>
</p>
</div>
<?php
   }
?>
</body>
</html>

<?php
   function auth_username() {
      return $_COOKIE['username'];
   }
   
   function auth_user_id() {
      return $_COOKIE['user_id'];
   }
  
  function home_url($additionalpath) {
    return 'http://'. $_SERVER['HTTP_HOST'].dirname($_SERVER['PHP_SELF']).'/'.$additionalpath;
  }
  
   if ( isset($_POST['submit']) and isset($_POST['logout_yes']) ) {
      setcookie('user_id', "", time() -1);
      setcookie('username', "", time() -1);
      header("Location: ". home_url("index.php") );
   }
   
   if ( empty($_COOKIE['user_id']) ) {
	 header("Location: ".home_url("index.php") );
   }
?>

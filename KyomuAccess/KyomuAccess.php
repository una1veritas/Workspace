<?php

class KyomuAccessPage {
	const serverPort = "tcp://localhost:3409";
	var $stream;
	var $session_id;
	var $user_id;
	var $login_id;
	var $error_no;
	var $error_msg;

	// Creates a new instance of this class which can be specified by web session id.
	function KyomuAccessPage() {
		$this->session_id = session_id();
	}

	// Tell delegate Java program to establish SSL connection to Kyomu server
	// with an appropriate "tool" assumed by id, and then login by uid and pwd.
	function open($uid, $pwd) {
		$msg = "";
		$this->stream = stream_socket_client(KyomuAccessPage::serverPort, $this->errno, $this->errmsg);
		if ($this->stream == false ) {
			$msg .= "Creating stream failed.";
			return $msg;
		}
		stream_set_timeout($this->stream, 10);
		$this->getResponces();
		fwrite($this->stream, "OPEN " . $this->session_id . "\r\n");
		$this->getResponces();
		fwrite($this->stream, "LOGIN " . $uid . " " . $pwd . "\r\n");
		$result = $this->getResponces();
		if ( preg_match("/^!SUCCEEDED LOGIN/", $result) == 1 ) {
			fwrite($this->stream, "status\r\n");
			$result = $this->getResponces();
			foreach (explode("\r\n", $result) as $line) {
				if ( preg_match("/^ACCESSID/", $line) > 0 ) {
					$tokens = explode("=",$line);
					$this->user_id = rtrim($tokens[1]);
				}
				if ( preg_match("/^LOGINID/", $line) > 0 ) {
					$tokens = explode("=",$line);
					$this->login_id = rtrim($tokens[1]);
				}
			}
			return $msg;
		}
		$msg .= "LOGIN failed w/ ".$result;
		return $msg;
	}

	function close() {
		stream_set_timeout($this->stream, 10);
		$this->getResponces();
		fwrite($this->stream, "OPEN " . $this->session_id . "\r\n");
		$this->getResponces();
		fwrite($this->stream, "QUIT" . "\r\n");
		$this->getResponces();
		return fclose($this->stream);
	}
	
	function test() {
		stream_set_timeout($this->stream, 10);
		$this->getResponces();
		fwrite($this->stream, "OPEN " . $this->session_id . "\r\n");
		$this->getResponces();
			fwrite($this->stream, "QUERY|MeiboTool|216#0:2011|205|" . "\r\n");
		$result = $this->getResponces();
		return $result;
	}

	// Gets a line terminated by end-of-line from this-stream.
	// This behaviour is completely different from readToken() in Java code.
	function getResponces() {
		stream_set_timeout($this->stream, 5);
		$resstr = "";
		while ($motd = fgets($this->stream)) {
			if ( preg_match("/^READY\./", $motd) > 0 )
			break;
			$resstr .= rtrim($motd) . "\r\n";
		}
		return $resstr;
	}

	function get_table($qstr) {
		$this->query($qstr);
		$result = array();
		for ( $rc = 0; $row = $this->next_row(); $rc++ ) {
			array_push($result, $row);
		}
		return $result;
	}

	function get_assoc($qstr) {
		$this->query($qstr);
		$result = array();
		while ( $row = $this->next_row() ) {
			$key = array_shift($row);
			$result[$key] = current($row);
		}
		return $result;
	}

	function get_firstrow($qstr) {
		$this->query($qstr);
		while ( $row = $this->next_row() ) {
			return $row;
		}
		return FALSE;
	}

}
?>

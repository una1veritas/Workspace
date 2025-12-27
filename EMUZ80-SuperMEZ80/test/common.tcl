set port /dev/cu.usbmodem1444301
catch { set port $env(PORT) }

set portid [open $port r+]
set default_timeout 30
set timeout $default_timeout
set send_slow { 1 .1 }

fconfigure $portid -mode "115200,n,8,1"
spawn -open $portid

expect_before {
    timeout { puts "\n\nTimeout detected"; exit 2 }
    eof     { puts "\n\nUnexpected EOD";   exit 1 }
}

proc open_cpm2 {} {
    # connect to the target
    # and reset the target to get sync
    send -break
    sleep 0.5
    send -s "reset\r"
    expect {
        "A>" {
            send "\r"
            expect "A>"
            send "reset\r"
        }
        "MON>" {
            send -s "\r"
            expect "MON>"
            send -s "reset\r"
        }
        -re "Select.*: " {
            send "\r"
        }
    }
    expect {
        "A>" { }
        -re "Select.*: " {
            send "\r"
            expect "A>"
        }
    }
}

proc open_cpm3 {} {
    # connect to the target
    # and reset the target to get sync
    send -break
    sleep 0.5
    send "reset\r"
    expect {
        "0: CPMDISKS.3" {
            set selection 0
        }
        "1: CPMDISKS.3" {
            set selection 1
        }
        "2: CPMDISKS.3" {
            set selection 2
        }
    }
    expect -re "Select.*: "
    send $selection
}

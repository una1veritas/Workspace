import os, sys, subprocess

if len(sys.argv) > 1 :
    subnet = sys.argv[1]
else:
    subnet = '192.168.1.1'
subnet = '.'.join(subnet.split('.')[:3])

if os.name != 'nt' :
    print('Sorry, this works only on Windows. ')    
    print('Try \'ssh pi@raspberrypi.local\' on macos.')
    exit(1)

cmd_ping = 'ping -n 1 -w 190 '
cmd_arp = 'arp -a '
    
print('host responces by ping command:')
for last_byte in range(1,255):
    try:
        ipnumber = (subnet + '.{0}').format(last_byte)
        #print(cmdstr)
        result = subprocess.run((cmd_ping+ipnumber).split(), shell=False, capture_output=True, check=True)
        #print(str(result.stdout.decode('sjis')))
        print(ipnumber + ' is active')
    except subprocess.CalledProcessError as err:
        #print(str(last_byte))
        pass

print('searching raspberry pi..')
try:
    result = subprocess.run(cmd_arp.split(), shell=False, capture_output=True, check=True)
    lines = [a_line.strip() for a_line in result.stdout.decode('sjis').split('\n')]
    lines = [a_line for a_line in lines if 'b8-27-eb' in a_line]
    if lines :
        for a_line in lines:
            macaddress = [a_token for a_token in (a_line.split()) if 'b8-27-eb' in a_token][0]
            ipaddress = [a_token for a_token in (a_line.split()) if subnet in a_token][0]
            
            print('raspberry pi ' + macaddress + '\'s ip-address = ' + ipaddress)
    else:
        print('no raspberry pi found.')
except subprocess.CalledProcessError as err:
    print(cmd_arp + ' error.')
    pass

print('bye.')
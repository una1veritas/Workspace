
cd ~/workspace/github/SuperMEZ80
if ! [ -e /opt/microchip/xc8/v2.36/bin/xc8 ]; then
    sudo ~/xc8-v2.36-full-install-linux-x64-installer.run
fi

echo "====================================="
echo "SuperMEZ80 build enviroment on ubuntu"
echo "====================================="
echo
echo "Try"
echo "    make realclean test_build"
echo "    or"
echo "    make BOARD=EMUZ80_57Q PIC=18F57Q43"
echo

if [ "$(cat ~/commands.sh)" == "" ]; then
    exec /bin/bash
else
    source ~/commands.sh
fi

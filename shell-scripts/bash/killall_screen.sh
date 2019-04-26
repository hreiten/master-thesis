source params.sh
sshpass -e ssh -t $user@129.241.211.$1 "killall screen; exit;"

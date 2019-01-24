source params.sh
ssh -N -L $1:localhost:$1 $user@$2

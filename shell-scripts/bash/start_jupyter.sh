source params.sh
screen_name=$default_screen_name
port=$default_port
computer=$default_computer
ip=$base_ip_banach.$computer

print_usage() {
        printf "\nError: invalid arguments \n"
	printf "Possible flags: -c: computer id\n"
	printf "\t\t-s: screen name \n"
	printf "\t\t-p: port number \n"
	printf "\t\t-u: base ip of computers in U2, default is banach\n"
	printf "\t\t-m: markov server\n\n"
	printf "Example usage: bash start_jupyter.sh -s 'name_of_screen' -p 'port_number' -c 'computer_id' \n\n"
}

i=1
argv=( "$@" )
while getopts ':c:p:s:u:m' flag; do
 case "${flag}" in
        c) computer=${argv[$i]} ;;
	s) screen_name=${argv[$i]} ;;
        p) port=${argv[$i]} ;;
        u) ip=$base_ip_u2.$computer ;;
	m) ip=$markov ;;
	*) print_usage
	   exit 1 ;;
 esac
 ((i++))
 ((i++))
done

printf "\n\t*********************************\n\t*\t\t\t\t*\n\t*"
printf "\tConfig:\t\t\t*\n\t*\t\t\t\t*\n\t*\tPort: $port\t\t*" 
printf "\n\t*\tScreen name: $screen_name\t*"
printf "\n\t*\tcomputer: $computer \t\t*\n\t*\t\t\t\t*\n\t*\t\t\t\t*"
printf "\n\t*********************************\n\n"

sshpass -e ssh -t $user@$ip \
"cd $path; bash shell-scripts/bash/create_screen.sh $screen_name; "\
"bash shell-scripts/bash/activate_venv.sh $screen_name; "\
"bash shell-scripts/bash/open_jupyter.sh $screen_name $port; "\
"exit; "\

bash jupyter_tunnel.sh $port $ip

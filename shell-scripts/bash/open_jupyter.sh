port=$(($port + $2))
printf "Opening port: $port \n"
screen -r $1 -X stuff $'jupyter notebook --no-browser --port='$port' --NotebookApp.token='' --NotebookApp.password=''\n'

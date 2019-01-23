### 
### Fish script to connect to a SSH, setup tmux, venv and Jupyter.
### Global parameters specified in params.fish.
###

# source global parameters from seperate file
#set DIR ()
source (dirname (status --current-filename))/params.fish

# if supplied arguments are less than 3, print usage instructions. 
# else, connect via SSH, source bash script stored there that will 
    # 1. connect to screen via tmux 
    # 2. activate virtual environment
    # 3. initiate jupyter notebook session with specified port
# then, a local jupyter tunnel is set up and activation links are printed to screen. 

if test (count $argv) -lt "3"
    echo "Usage >> source connect-with-jupyter.fish COMPUTERID PORT TMUXNAME"
    echo "Example >> source connect-with-jupyter.fish 60 8080 myname"
    echo "Important that the arguments comes exactly as listed!"
    echo "Nullrommet (15 - 45?), Banach (45 - 70?)"

else 
    set computerID $argv[1];
    set port $argv[2];
    set tmux_name $argv[3];
    set ip $baseip.$computerID

    echo -e "Specified IP:       \t $ip"
    echo -e "Specified port:     \t $port"
    echo -e "Specified tmux name:\t $tmux_name \n"

    # call script from ssh that sets up tmux, activates venv and initiates jupyter
    ssh -X -C $user@$ip "source setup-jupyter-with-tmux.sh -p $port -n $tmux_name"

    # set up Jupyter tunnel locally
    echo "Setting up Jupyter tunnel at port: $port"
    ssh -N -f -L $port:localhost:$port $user@$ip

    # display Jupyter sessions
    echo -e "\nRetrieving list of active Jupyter sessions..."
    ssh -X -C $user@$ip "jupyter notebook list; echo -e '\nActive screens on IP $ip:'; tmux ls"
end


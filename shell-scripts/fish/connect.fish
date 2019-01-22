### 
### Fish script to connect to a SSH. 
### Parameters specified in params.fish.
###

# source global parameters from seperate file
set DIR (cd (dirname (status -f)); and pwd)
source "$DIR"/params.fish

# if zero arguments given, print instructions
# else, set computerID to given argument and connect via SSH
if test (count $argv) = 0
  echo "Usage >> source connect.fish COMPUTERID"
  echo "Example >> source connect.fish 60"
else
  set computerID $argv[1];
  set ip $baseip.$computerID;
  ssh -X -C $user@$ip;
end


### 
### Fish script to kill all active tmux screens at a given IP. 
### Global parameters specified in params.fish.
###

# source global parameters from params.fish
set DIR (dirname (status --current-filename))
source $PWD/$DIR/params.fish

if test (count $argv) = 0
  echo "Usage >> source kill-sessions.fish COMPUTERID"
  echo "Example >> source kill-sessions.fish 60"
  echo "User confirmation required to kill all tmux sessions."
else
  set computerID $argv[1];
  set ip $baseip.$computerID;

  set tmp 0
  function read_confirm
    while true
      read -l -P 'Continue? [y/N] ' confirm

      switch $confirm
        case Y y
          set tmp 1
          break
        case '' N n
          set tmp 0
          break
      end
    end
  end

  ssh -X -C $user@$ip "echo -e '\nActive tmux screens:'; tmux ls; jupyter notebook list"

  read_confirm
  if test "$tmp" = 1
      ssh -X -C $user@$ip "tmux kill-server"
      echo "Active sessions terminated!"
  end
end



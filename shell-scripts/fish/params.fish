set user halvorre
set baseip 129.241.211

if not set -q computerID:
    set computerID 60
    set ip $baseip.$computerID
end

if not set -q port: 
    set port 8080
end
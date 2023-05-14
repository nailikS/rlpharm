set q=%1
set o=%2
set a=%3
set i=%4
cd "C:\Users\kilia\MASTER\rlpharm\LS5"
wsl bash -c "C:\Users\kilia\MASTER\rlpharm\LS5\iscreen.sh --query %q% --database %i%, %a% --output %o%"
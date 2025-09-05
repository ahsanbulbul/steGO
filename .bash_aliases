# +------------------------------------+ #
# |        [Custom bash_aliases]       | #
# |                                    | #
# |           System: debian           | #
# |         version: bookworm          | #
# |         Version: 10 JUN 24         | #
# +------------------------------------+ #

alias la='ls -lah'
alias ..='cd ..'
alias py='python3'
alias dots='/usr/bin/git --git-dir=/home/mirage/.dots.git/ --work-tree=/'
alias dotfiles='git --git-dir=$HOME/.dots.git ls-tree -r --name-only HEAD'
alias dist='/usr/bin/git --git-dir=$HOME/.distros/.recepie/.git --work-tree=$HOME/.distros/'
alias warpgrade='sudo apt update; sudo apt upgrade -y'
alias sleepOFF='sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target'
alias sleepON='sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target'
alias onyx='source ~/.onyx/bin/activate'
alias ct='wl-copy'
alias pt='wl-paste'
alias enterVoid='sshfs azul@avoid:/core/ddata /server/'
ssh() {
	if [[ "$1" == "void" ]]; then
		network=$(/sbin/iwgetid -r)
		if [[ "$network" == "WiFire" ]]; then
			command ssh avoid "${@:2}"
		else
			command ssh hypervoid "${@:2}"
		fi
	else
		command ssh "$@"
	fi
}

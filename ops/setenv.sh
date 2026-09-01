# setenv for the try_pheno container. Source this from inside the container:
#   source /app/ops/setenv.sh
export PYDEVD_WARN_EVALUATION_TIMEOUT=1000000
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=1000000
export HF_HUB_OFFLINE=1
export PROJ_NM=try_pheno
export PROJ_V=`git branch --show-current`
export PROJ_NMV="${PROJ_NM}-${PROJ_V}"

export opsdir=$( cd "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd )
export rootdir=$( cd "${opsdir}/.." && pwd )

list_commands () {
	echo COMMANDS:
	echo	list_commands
	echo 	::: ${PROJ_NMV} :::
	echo	source /app/ops/setenv.sh
}

list_commands

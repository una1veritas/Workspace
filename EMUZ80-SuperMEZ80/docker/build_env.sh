#!/bin/bash

export LANG=en_US.UTF-8

USER=user
IMAGENAME=build-supermez80
CONTAINERNAME=${IMAGENAME}-container

main()
{
    setup

    i=1
    args=$#
    while [ $i -le $args ]; do
        case $1 in
        clean)
            remove_image
            exit 0
            ;;
        --)
            shift
            break
            ;;
        --clean)
            remove_image
            shift
            ;;
        *)
            break
            ;;
        esac
    done

    if [ "$*" != "" ]; then
        echo $* | tr -d '\r' > ${SCRIPT_DIR}/commands.sh
        echo Run \'$(cat commands.sh)\' in build environment
    else
        echo > ${SCRIPT_DIR}/commands.sh
    fi

    IMGID=$(image_id)
    if [ "$IMGID" == "" ]; then
        build_image
    fi
    CNTRID=$(container_id)
    if [ "$CNTRID" == "" ]; then
        run_image
    else
        docker start $CNTRID
        docker attach $CNTRID
    fi
}

setup()
{
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    echo SCRIPT_DIR=${SCRIPT_DIR}
    cd ${SCRIPT_DIR}
    
    case $(uname -s) in
    Linux|Darwin)
        WINPTY=
        WD=${SCRIPT_DIR}
        ;;
    MINGW*)
        WINPTY=winpty
        HOME_mod=$(echo $USERPROFILE | sed -e 's|:\\|://|' -e 's|\\|/|g')
        WD=$(echo ${SCRIPT_DIR} | sed -e "s|^${HOME}|${HOME_mod}|")
        HOME=$HOME_mod
        ;;
    *)
        echo Can not run on $(uname -s)
        exit 1
        ;;
    esac

    echo HOME=${HOME}
    echo WD=${WD}

    tr -d '\r' < ./run.sh > ./run_mod.sh && chmod +x ./run_mod.sh
    tr -d '\r' < ./dot_bashrc > ./dot_bashrc_mod
}

cleanup_containers()
{
    id=$( docker ps -a --filter ancestor=${IMAGENAME} | grep -v -e '^CONTAINER ID' | awk '{ print $1 }' )
    if [ "${id}" != "" ]; then
        docker stop ${id}
        docker rm ${id}
    fi
}

remove_image()
{
    cleanup_containers
    docker rmi ${IMAGENAME}
}

image_id()
{
    docker image ls | grep -e "^${IMAGENAME}" | awk '{print $3}'
}

container_id()
{
    docker ps -a --filter name=${CONTAINERNAME} | grep -v "^CONTAINER ID" | awk '{print $1}'
}

build_image()
{
    docker build --network=host --file Dockerfile \
           --build-arg="UID=$(id -u)" \
           --build-arg="GID=$(id -g)" \
           --build-arg="USER=${USER}" \
           --tag ${IMAGENAME} .
}

run_image()
{
    ${WINPTY} docker run  --privileged \
       --name ${CONTAINERNAME} -it \
       --volume="${HOME}/.netrc:/home/user/.netrc:ro" \
       --volume="${HOME}/.ssh:/home/user/.ssh:ro" \
       --volume="${WD}/dot_bashrc_mod:/home/user/.bashrc:rw" \
       --volume="${WD}/run_mod.sh:/home/user/run.sh:rw" \
       --volume="${WD}/commands.sh:/home/user/commands.sh:rw" \
       --volume="${WD}/..:/home/user/workspace/github/SuperMEZ80:rw" \
       ${IMAGENAME} \
       //bin/bash -c ./run.sh
}

main $*

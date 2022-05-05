#!/bin/bash

set -eu
unset CDPATH
cd "$( dirname "$0" )/../.."

USAGE="
Usage:
    $0 [OPTION...] [[--] TEST_ARGS...]
Run tests in a docker or singularity container.
Options:
    -h, --help         Print this help and exit
    -B, --no-build     Don't build the docker image (use existing)
    -C, --no-cache     Give Docker the --no-cache argument
    -s, --shell        Drop into the container with bash instead of normal entry
    -S, --singularity  Run in Singularity container instead of Docker
    -- TEST_ARGS       Arguments passed to tests.sh
"

main() {
    BUILD_IMAGE=1
    CACHE=""
    DOCKERFILE="tests/Dockerfile"
    DOCKER_TAG="testing"
    WORK_DIR="-w=/flywheel/v0"
    ENTRY_POINT="--entrypoint=tests/bin/run_pytest.sh"
    #ENTRY_POINT="--entrypoint=/flywheel/v0/tests/bin/poetry_test.sh"

    SINGULARITY_CMD="run"
    RUN="Docker"
    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help)
                printf "$USAGE" >&2
                exit 0
                ;;
            -B|--no-build)
                BUILD_IMAGE=
                ;;
            -C|--no-cache)
		CACHE="--no-cache"
                ;;
            -s|--shell)
                ENTRY_POINT="--entrypoint=/bin/bash"
                SINGULARITY_CMD="shell"
                ;;
            -S|--singularity)
                RUN="Singularity"
                ;;
            --)
                shift
                break
                ;;
            *)
                break
                ;;
        esac
        shift
    done

    VER=$(cat manifest.json | jq -r '.version')
    DOCKER_IMAGE_NAME=$(cat manifest.json | jq '.custom."gear-builder".image' | tr -d '"')
    echo "DOCKER_IMAGE_NAME is" $DOCKER_IMAGE_NAME

    MANIFEST_NAME=$(cat manifest.json | jq '.name'  | tr -d '"')
    TESTING_IMAGE="flywheel/${MANIFEST_NAME}:${DOCKER_TAG}"
    echo "TESTING_IMAGE is $TESTING_IMAGE"


    if [ "${BUILD_IMAGE}" = "1" ]; then

	set -x
        docker build -f Dockerfile -t "${DOCKER_IMAGE_NAME}" .

        docker build -f "${DOCKERFILE}" \
          --build-arg DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME} \
          -t "${TESTING_IMAGE}" .
	set +x

        if [ "$RUN" = "Singularity" ]; then
            if [ -e ${MANIFEST_NAME}.sif ]; then
                rm ${MANIFEST_NAME}.sif
            fi
	    set -x
            singularity build ${MANIFEST_NAME}.sif docker-daemon://${TESTING_IMAGE}
	    set +x
        fi

    fi


    echo "Running in a $RUN container"
    if [ "$RUN" = "Docker" ]; then
	set -x
        docker run "${WORK_DIR}" -it --rm \
            --volume "$HOME/.config/flywheel:/root/.config/flywheel" \
            "${ENTRY_POINT}" \
            "${TESTING_IMAGE}" \
            "$@"
	set +x

    else
	set -x
        singularity ${SINGULARITY_CMD} ${MANIFEST_NAME}.sif
	set +x
    fi

}

main "$@"

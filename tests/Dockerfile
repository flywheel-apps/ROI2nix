ARG DOCKER_IMAGE_NAME

FROM ${DOCKER_IMAGE_NAME}

RUN poetry install

COPY tests ${FLYWHEEL}/tests
RUN chmod +x ${FLYWHEEL}/tests/bin/run_tests.sh
ENV PATH=/flywheel/v0/converters/slicer/slicer:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENTRYPOINT ["/flywheel/v0/tests/bin/run_pytest.sh"]



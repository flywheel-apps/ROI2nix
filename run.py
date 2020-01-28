#!/usr/bin/env python3
import os, os.path as op
import json
import subprocess as sp
import copy
import shutil
import logging

import flywheel

if __name__ == '__main__':
    # Get the Gear Context
    context = flywheel.GearContext()

    # Activate custom logger
    log_name = '[roi2nix]'
    log_level = logging.INFO
    fmt = '%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s %(funcName)s()]: %(message)s'
    logging.basicConfig(level=log_level, format=fmt, datefmt='%H:%M:%S')
    context.log = logging.getLogger(log_name)
    context.log.critical('{} log level is {}'.format(log_name, log_level))

    context.log_config()

    # Build, Validate, and execute Parameters Hello World 
    try:
        # build the command string
        command = ['echo']
        for key in context.config.keys():
            command.append(context.config[key])
        
        # execute the command string
        result = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE,
                    universal_newlines=True)
        stdout, stderr = result.communicate()
        context.log.info('Command return code: {}'.format(result.returncode))

        context.log.info(stdout)

        if result.returncode != 0:
            raise Exception(stderr)
        
    except Exception as e:
        context.log.fatal(e,)
        context.log.fatal(
            'Error executing roi2nix.',
        )
        os.sys.exit(1)

    context.log.info("roi2nix completed Successfully!")
    os.sys.exit(0)
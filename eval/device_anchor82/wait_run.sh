#!/bin/bash
# wait until a file exists (e.g. a manifest still being built), then run the command
f=$1; shift
until [ -f "$f" ]; do sleep 15; done
"$@"

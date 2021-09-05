#!/bin/sh
# We are telling Gunicorn to spawn 2 worker processes, running 2 threads each.
# app is name of file: app is specific flask variable
# We are also accepting connections from outside, and overriding Gunicorn's default port (8000).
# make sure EOL Sequence is LF (VSCode as drop down) if get error: standard_init_linux.go:190: exec user process caused "no such file or directory"
gunicorn app:app -w 2 --threads 2 -b 0.0.0.0:8003
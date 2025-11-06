#!/usr/bin/env bash

# Command to run the Gunicorn production server, 
# hosting your Flask application defined in app.py
gunicorn --bind 0.0.0.0:$PORT app:app
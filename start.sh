#!/bin/bash
# démarrer uvicorn sur le port fourni par Railway
uvicorn diabetes:app --host 0.0.0.0 --port $PORT

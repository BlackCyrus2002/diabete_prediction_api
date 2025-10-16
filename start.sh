#!/bin/bash
# démarrer uvicorn sur le port fourni par Railway
uvicorn iris:app --host 0.0.0.0 --port $PORT

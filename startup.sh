#!/usr/bin/env bash
#Dockerfileでは複数コマンド実行できないためにシェルで複数コマンド実行。
cd /workspace/ai-app-for-docker
streamlit run streamlit.py
#streamlit run streamlit.py --server.enableCORS=false
#streamlit run streamlit.py --server.enableWebsocketCompression=false
#streamlit run my_app.py --server.enableXsrfProtection=false

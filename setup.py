#!/usr/bin/env python
import os, pathlib
home = str(pathlib.Path.home())

os.makedirs(home+"/.streamlit", exist_ok=True)
with open(home+"/.streamlit/credentials.toml", "w") as fp:
    fp.write("[general]\nemail = \"your@domain.com\"\n")
with open(home+"/.streamlit/config.toml", "w") as fp:
    fp.write("[server]\nheadless = true\nenableCORS=false\nport = $PORT\n")
with open(home+"/.heroku/python/lib/python3.7/site-packages/streamlit/static/index.html", "r+") as fp:
    txt = fp.read()
    txt2 = txt.replace("<head>", "<head><script async src=\"https://www.googletagmanager.com/gtag/js?id=G-YV3ZFR8VG6\"></script><script>window.dataLayer = window.dataLayer || [];function gtag(){dataLayer.push(arguments);}gtag('js', new Date());gtag('config', 'G-YV3ZFR8VG6');</script>")
    fp.seek(0)
    fp.write(txt2)
    fp.truncate()

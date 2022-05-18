#!/usr/bin/env python
import os, pathlib
home = str(pathlib.Path.home())

os.makedirs(home+"/.streamlit", exist_ok=True)
with open(home+"/.streamlit/credentials.toml", "w") as fp:
    fp.write("[general]\nemail = \"your@domain.com\"\n")
with open(home+"/.streamlit/config.toml", "w") as fp:
    fp.write(f"[server]\nheadless = true\nenableCORS=false\nport = {os.path.expandvars('$PORT')}\n")
    fp.write(f"[runner]\nfastReruns = true\n")

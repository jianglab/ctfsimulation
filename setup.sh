mkdir -p ~/.streamlit/
echo '
[general]
email = ”your-email@domain.com”
' > ~/.streamlit/credentials.toml
echo '
[server]
headless = true
enableCORS=false
port = $PORT
' > ~/.streamlit/config.toml
# import os
# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ['REQUESTS_CA_BUNDLE'] = ''

# import requests
# from huggingface_hub import configure_http_backend

# def backend_factory() -> requests.Session:
#     session = requests.Session()
#     session.verify = False
#     return session

# configure_http_backend(backend_factory=backend_factory)

def main() -> None:
    print("Hello from arc-tartiflette!")

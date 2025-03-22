from waitress import serve
from deploy import app

if __name__ == "__main__":
    print('Server Started')
    serve(
        app,
        host='0.0.0.0',
        port=8080,
    )
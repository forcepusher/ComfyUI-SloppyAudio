import subprocess
import sys
import os


def install_requirements():
    req = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", req]
        )


if __name__ == "__main__":
    install_requirements()

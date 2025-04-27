import subprocess
import sys

def install_ffmpeg():
    print("Starting ffmpeg installation...")

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"])

    # install python library
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
        print("Installed ffmpeg-python completely")
    except subprocess.CalledProcessError as e:
        print("Failed to install ffmpeg-python")
        print(e)

    # download binary file
    try:
        subprocess.check_call([
            "wget",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "-O", "/tmp/ffmpeg.tar.xz"
        ])

        subprocess.check_call(["tar", "-xf", "/tmp/ffmpeg.tar.xz", "-C", "/tmp/"])

        # Find the location path to ffmpeg file inside /temp/ directory
        ffmpeg_path = subprocess.run(
            ["find", "/tmp", "-name", "ffmpeg", "-type", "f"],
            capture_output=True,
            text=True
        ).stdout.strip() # print to terminal

        # make binary globally accessible
        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])
        
        # Make binary executable
        subprocess.check_call("chmod", "+x", "/usr/local/bin/ffmpeg")

        print("Intalled static ffmpeg binary successfully")
    except Exception as e:
        print(f"Failed to install static ffmpeg binary: {e}")

    # Verify installation
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        print("ffmpeg verion:")
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ffmpeg installation verification failed.")
        return False
    
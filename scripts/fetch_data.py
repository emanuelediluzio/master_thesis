import subprocess
import os

HOST = "ediluzio@ailb-login-03.ing.unimore.it"
PASSWORD = "Fulmine88!\n"
HTML_PATH = "/work/tesi_ediluzio/inferenza/visual_benchmark_30samples/html/multi_model_comparison_cherry30.html"
IMG_DIR = "/work/tesi_ediluzio/inferenza/visual_benchmark_30samples/images/"

def fetch_html():
    print("Fetching HTML...")
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", HOST, f"cat '{HTML_PATH}'"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate(input=PASSWORD.encode())
    
    # The output will contain "password:" prompt and maybe welcome text?
    # SSH usually prints prompt to tty, but sometimes stderr.
    # We'll save all stdout.
    
    # Filter out potential password prompt if it got into stdout (unlikely, usually stderr/tty)
    content = out.decode('utf-8', errors='ignore')
    # Find start of HTML
    start_idx = content.find("<!DOCTYPE html>")
    if start_idx != -1:
        content = content[start_idx:]
    
    with open("cherry30.html", "w") as f:
        f.write(content)
    print(f"HTML saved ({len(content)} bytes).")

def fetch_images():
    print("Fetching Images 00-15...")
    images = [f"sample_{i:02d}.png" for i in range(22)] # Get 00 to 21
    img_list = " ".join(images)
    remote_cmd = f"cd {IMG_DIR} && tar czf - {img_list}"
    
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", HOST, remote_cmd]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate(input=PASSWORD.encode())
    
    if len(out) > 0:
        with open("images.tar.gz", "wb") as f:
            f.write(out)
        print(f"Images tarball saved ({len(out)} bytes).")
        # Untar
        subprocess.run(["tar", "xzf", "images.tar.gz"])
        # Move to figures/ (renaming sample_00.png to sample_0.png if needed, or keeping it)
        # We will check filenames later.
    else:
        print("Error fetching images:", err.decode())

if __name__ == "__main__":
    fetch_html()
    fetch_images()

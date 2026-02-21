import subprocess
import time

HOST = "ediluzio@ailb-login-03.ing.unimore.it"
PASSWORD = "Fulmine88!\n"
RAW_HTML = "/work/tesi_ediluzio/inferenza/visual_benchmark_30samples/html/multi_model_comparison_cherry30.html"
CLEAN_HTML = "/tmp/clean_cherry30.html"
OUTPUT_FILE = "extracted_metadata.txt"

def run_ssh(cmd, description):
    print(f"Running: {description}")
    full_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", HOST, cmd]
    p = subprocess.Popen(full_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate(input=PASSWORD.encode())
    return out.decode('utf-8', errors='ignore'), err.decode('utf-8', errors='ignore')

def main():
    # 1. Create clean file on server
    create_cmd = f"sed 's/<img src[^>]*>//g' {RAW_HTML} > {CLEAN_HTML}"
    out, err = run_ssh(create_cmd, "Creating clean HTML on server")
    if err and "password" not in err.lower():
        print(f"Error creating file: {err}")

    # 2. Loop and fetch
    with open(OUTPUT_FILE, "w") as f:
        for i in range(16):
            # Grep for Sample ID and following lines (covers GT and Table)
            # We grep for the ID, and then -A 20 to get the content
            grep_cmd = f"grep -A 25 \"id='sample-{i}'\" {CLEAN_HTML}"
            content, err = run_ssh(grep_cmd, f"Fetching Sample {i}")
            
            f.write(f"--- SAMPLE {i} ---\n")
            f.write(content)
            f.write("\n\n")
            time.sleep(0.5)

    print(f"Done. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

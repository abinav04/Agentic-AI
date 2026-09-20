import subprocess
import sys
import time
import os
import signal
import socket

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def free_port(port: int):
    """Frees specified port if it is currently occupied by a stale process."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        res = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        if res == 0:
            print(f"Clearing stale background process on port {port}...")
            if os.name == 'nt':
                cmd = f'powershell -Command "Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"'
                subprocess.run(cmd, shell=True, capture_output=True)
            time.sleep(1.0)
    except Exception:
        pass

def main():
    print("=" * 60)
    print(" Kestrel Labs Multi-Agent Assistant - One-Command Launcher")
    print("=" * 60)
    
    # Free ports 8000 and 8501 before starting
    free_port(8000)
    free_port(8501)

    python_exe = sys.executable

    backend_cmd = [python_exe, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"]
    frontend_cmd = [python_exe, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501"]

    processes = []

    try:
        # Start FastAPI Backend
        print("Starting FastAPI Backend on http://localhost:8000 ...")
        backend_proc = subprocess.Popen(backend_cmd)
        processes.append(("FastAPI Backend", backend_proc))
        
        # Wait a moment for backend to initialize
        time.sleep(2.5)

        # Start Streamlit Frontend
        print("Starting Streamlit UI Frontend on http://localhost:8501 ...")
        frontend_proc = subprocess.Popen(frontend_cmd)
        processes.append(("Streamlit UI", frontend_proc))

        print("\n" + "=" * 60)
        print(" Both services are running!")
        print(" Backend API Docs : http://localhost:8000/docs")
        print(" Streamlit UI     : http://localhost:8501")
        print(" Press Ctrl+C to terminate both servers safely.")
        print("=" * 60 + "\n")

        # Monitor processes
        while True:
            time.sleep(1)
            for name, proc in processes:
                poll = proc.poll()
                if poll is not None:
                    print(f"Warning: Process '{name}' terminated unexpectedly with exit code {poll}.")
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\nShutting down backend and frontend services...")
        for name, proc in processes:
            if proc.poll() is None:
                print(f"  Stopping {name}...")
                if os.name == 'nt':
                    proc.send_signal(signal.CTRL_BREAK_EVENT if hasattr(signal, 'CTRL_BREAK_EVENT') else signal.SIGTERM)
                    proc.kill()
                else:
                    proc.terminate()
        print("Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()

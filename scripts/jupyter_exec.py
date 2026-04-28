#!/usr/bin/env python3
"""HTTPS-proxied command executor for RunPod pods (when port 22 is firewalled).

Drives a Jupyter kernel over the pod's HTTPS proxy URL to execute arbitrary
shell/Python commands. Returns stdout/stderr from the command.

Usage:
    python3 jupyter_exec.py --pod-id <id> --password <pw> shell "ls -la"
    python3 jupyter_exec.py --pod-id <id> --password <pw> upload <local> <remote>
    python3 jupyter_exec.py --pod-id <id> --password <pw> download <remote> <local>
    python3 jupyter_exec.py --pod-id <id> --password <pw> bg-start <cmd> --tag <name>
    python3 jupyter_exec.py --pod-id <id> --password <pw> bg-tail <name> [--lines N]

Env: POD_ID, POD_PASSWORD override flags.
"""
import argparse, base64, json, os, sys, time, uuid
import requests, websocket


def proxy_url(pod_id, port=8888):
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def login(pod_id, password):
    s = requests.Session()
    base = proxy_url(pod_id)
    r = s.get(f"{base}/login")
    r.raise_for_status()
    xsrf = s.cookies.get("_xsrf")
    if not xsrf:
        sys.exit(f"no _xsrf cookie returned by {base}/login")
    r = s.post(f"{base}/login", data={"_xsrf": xsrf, "password": password}, allow_redirects=False)
    if r.status_code not in (302, 200):
        sys.exit(f"login failed: HTTP {r.status_code}")
    if not s.cookies.get("username-29njhkoej7hejo-8888".replace("29njhkoej7hejo", pod_id), None):
        # Jupyter sets a session cookie of varying name; existence check is fragile, skip.
        pass
    return s, base


def _ws_url(base, kernel_id, session_id, xsrf):
    return base.replace("https://", "wss://") + f"/api/kernels/{kernel_id}/channels?session_id={session_id}"


def _start_kernel(s, base):
    r = s.post(f"{base}/api/kernels", headers={"X-XSRFToken": s.cookies.get("_xsrf")})
    r.raise_for_status()
    return r.json()["id"]


def _delete_kernel(s, base, kernel_id):
    try:
        s.delete(f"{base}/api/kernels/{kernel_id}", headers={"X-XSRFToken": s.cookies.get("_xsrf")})
    except Exception:
        pass


def _exec_code(s, base, code, timeout=600):
    """Execute python code in a fresh kernel; collect stdout/stderr/results.

    Returns (stdout_str, stderr_str, status, exec_count).
    """
    kernel_id = _start_kernel(s, base)
    session_id = uuid.uuid4().hex
    msg_id = uuid.uuid4().hex
    cookie_header = "; ".join(f"{k}={v}" for k, v in s.cookies.items())
    ws_url = _ws_url(base, kernel_id, session_id, s.cookies.get("_xsrf"))
    ws = websocket.create_connection(
        ws_url,
        header=[f"Cookie: {cookie_header}"],
        timeout=timeout,
    )
    try:
        msg = {
            "header": {
                "msg_id": msg_id,
                "username": "claude",
                "session": session_id,
                "msg_type": "execute_request",
                "version": "5.3",
                "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "channel": "shell",
        }
        ws.send(json.dumps(msg))
        stdout, stderr, status = [], [], "unknown"
        deadline = time.time() + timeout
        while time.time() < deadline:
            ws.settimeout(min(30, deadline - time.time()))
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if not raw:
                break
            d = json.loads(raw)
            parent = d.get("parent_header", {}).get("msg_id")
            if parent != msg_id:
                continue
            mtype = d.get("msg_type")
            content = d.get("content", {})
            if mtype == "stream":
                if content.get("name") == "stdout":
                    stdout.append(content.get("text", ""))
                else:
                    stderr.append(content.get("text", ""))
            elif mtype == "error":
                stderr.append("\n".join(content.get("traceback", [])))
                status = "error"
            elif mtype == "execute_reply":
                status = content.get("status", status)
            elif mtype == "status" and content.get("execution_state") == "idle":
                # Idle after our request means execution finished.
                if d.get("parent_header", {}).get("msg_type") == "execute_request":
                    break
        return "".join(stdout), "".join(stderr), status
    finally:
        try:
            ws.close()
        except Exception:
            pass
        _delete_kernel(s, base, kernel_id)


def cmd_shell(s, base, command, timeout=600, cwd=None):
    cwd_prefix = f"cd {json.dumps(cwd)} && " if cwd else ""
    code = (
        "import subprocess, sys\n"
        f"r = subprocess.run({json.dumps(cwd_prefix + command)}, shell=True, executable='/bin/bash', "
        "capture_output=True, text=True)\n"
        "sys.stdout.write(r.stdout)\n"
        "sys.stderr.write(r.stderr)\n"
        "print('__EXIT__:%d' % r.returncode)\n"
    )
    out, err, st = _exec_code(s, base, code, timeout=timeout)
    rc = -1
    if "__EXIT__:" in out:
        rc_line = out.rsplit("__EXIT__:", 1)[1].split("\n", 1)[0].strip()
        try:
            rc = int(rc_line)
        except Exception:
            pass
        out = out.rsplit("__EXIT__:", 1)[0]
    return out, err, rc


def cmd_upload(s, base, local_path, remote_path):
    with open(local_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    code = (
        "import base64, os\n"
        f"data = base64.b64decode({json.dumps(b64)})\n"
        f"path = {json.dumps(remote_path)}\n"
        "os.makedirs(os.path.dirname(path) or '.', exist_ok=True)\n"
        "with open(path, 'wb') as f:\n"
        "    f.write(data)\n"
        f"print('uploaded', path, len(data), 'bytes')\n"
    )
    out, err, st = _exec_code(s, base, code, timeout=300)
    return out, err, st


def cmd_download(s, base, remote_path, local_path):
    code = (
        "import base64, os, sys\n"
        f"path = {json.dumps(remote_path)}\n"
        "if not os.path.exists(path):\n"
        "    print('__MISSING__')\n"
        "else:\n"
        "    with open(path, 'rb') as f:\n"
        "        sys.stdout.write(base64.b64encode(f.read()).decode())\n"
    )
    out, err, st = _exec_code(s, base, code, timeout=600)
    if "__MISSING__" in out:
        sys.exit(f"remote file not found: {remote_path}")
    raw = base64.b64decode(out.strip())
    os.makedirs(os.path.dirname(os.path.abspath(local_path)) or ".", exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(raw)
    return f"downloaded {remote_path} -> {local_path} ({len(raw)} bytes)", err, st


def cmd_bg_start(s, base, command, tag, cwd=None):
    cwd_prefix = f"cd {json.dumps(cwd)} && " if cwd else ""
    log_path = f"/workspace/_bg_{tag}.log"
    pid_path = f"/workspace/_bg_{tag}.pid"
    inner = cwd_prefix + command
    code = (
        "import subprocess, os, json, signal\n"
        f"log = {json.dumps(log_path)}\n"
        f"pid = {json.dumps(pid_path)}\n"
        f"cmd = {json.dumps(inner)}\n"
        "os.makedirs('/workspace', exist_ok=True)\n"
        "with open(log, 'w') as f: pass\n"
        "p = subprocess.Popen(['/bin/bash', '-lc', cmd], stdout=open(log,'w'), stderr=subprocess.STDOUT, "
        "preexec_fn=os.setsid)\n"
        "with open(pid, 'w') as f: f.write(str(p.pid))\n"
        f"print(json.dumps({{'pid': p.pid, 'log': log, 'tag': {json.dumps(tag)}}}))\n"
    )
    return _exec_code(s, base, code, timeout=60)


def cmd_bg_tail(s, base, tag, lines=200):
    log_path = f"/workspace/_bg_{tag}.log"
    pid_path = f"/workspace/_bg_{tag}.pid"
    code = (
        "import os, subprocess\n"
        f"log = {json.dumps(log_path)}\n"
        f"pid_file = {json.dumps(pid_path)}\n"
        f"n = {int(lines)}\n"
        "alive = False\n"
        "if os.path.exists(pid_file):\n"
        "    pid = int(open(pid_file).read().strip())\n"
        "    try:\n"
        "        os.kill(pid, 0); alive = True\n"
        "    except ProcessLookupError:\n"
        "        alive = False\n"
        "print(f'__ALIVE__={alive}')\n"
        "if os.path.exists(log):\n"
        "    out = subprocess.run(['tail','-n',str(n),log], capture_output=True, text=True).stdout\n"
        "    print(out, end='')\n"
        "else:\n"
        "    print('(no log yet)')\n"
    )
    return _exec_code(s, base, code, timeout=60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod-id", default=os.environ.get("POD_ID"))
    ap.add_argument("--password", default=os.environ.get("POD_PASSWORD", "parameter-golf"))
    ap.add_argument("--cwd", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("shell"); sp.add_argument("command")
    sp = sub.add_parser("upload"); sp.add_argument("local"); sp.add_argument("remote")
    sp = sub.add_parser("download"); sp.add_argument("remote"); sp.add_argument("local")
    sp = sub.add_parser("bg-start"); sp.add_argument("command"); sp.add_argument("--tag", required=True)
    sp = sub.add_parser("bg-tail"); sp.add_argument("tag"); sp.add_argument("--lines", type=int, default=200)
    args = ap.parse_args()

    if not args.pod_id:
        sys.exit("--pod-id or POD_ID env required")
    s, base = login(args.pod_id, args.password)
    if args.cmd == "shell":
        out, err, rc = cmd_shell(s, base, args.command, timeout=args.timeout, cwd=args.cwd)
        sys.stdout.write(out)
        if err:
            sys.stderr.write(err)
        sys.exit(rc)
    elif args.cmd == "upload":
        out, err, st = cmd_upload(s, base, args.local, args.remote)
        print(out)
        sys.exit(0 if st == "ok" else 1)
    elif args.cmd == "download":
        out, err, st = cmd_download(s, base, args.remote, args.local)
        print(out)
        sys.exit(0 if st == "ok" else 1)
    elif args.cmd == "bg-start":
        out, err, st = cmd_bg_start(s, base, args.command, args.tag, cwd=args.cwd)
        print(out)
        sys.exit(0 if st == "ok" else 1)
    elif args.cmd == "bg-tail":
        out, err, st = cmd_bg_tail(s, base, args.tag, lines=args.lines)
        print(out)
        sys.exit(0 if st == "ok" else 1)


if __name__ == "__main__":
    main()

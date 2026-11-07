import json
import subprocess
import sys


def run(*a):
    return subprocess.run([sys.executable, "-m", "contextrag", *a], capture_output=True, text=True)


def test_keys_list_and_revoke_never_print_secrets(tmp_path):
    f = tmp_path / "keys.json"
    key = run("keygen", "--id", "ana", "--tenant", "acme", "--add-to", str(f)).stdout.strip()
    run("keygen", "--id", "lia", "--tenant", "acme", "--add-to", str(f))
    out = run("keys", "list", "--file", str(f))
    assert out.returncode == 0 and key not in out.stdout and "crag_" not in out.stdout
    assert {json.loads(x)["id"] for x in out.stdout.splitlines()} == {"ana", "lia"}
    r = run("keys", "revoke", "--file", str(f), "--id", "ana")
    assert r.returncode == 0 and json.loads(r.stdout) == {"revoked": 1}
    assert [e["id"] for e in json.loads(f.read_text()).values()] == ["lia"]
    assert run("keys", "revoke", "--file", str(f), "--id", "nobody").returncode == 1
    assert run("keys", "revoke", "--file", str(f)).returncode == 2

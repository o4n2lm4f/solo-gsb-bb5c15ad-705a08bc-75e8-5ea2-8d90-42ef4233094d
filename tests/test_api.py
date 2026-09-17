"""End-to-end tests. Run with: pytest -q"""

import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

# Point the DB at a throwaway file before importing the app.
_tmp = tempfile.mkdtemp()
os.environ["CIDR_DB_PATH"] = str(Path(_tmp) / "test.db")

from fastapi.testclient import TestClient  # noqa: E402

from app.db import connect, init_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture()
def client():
    # Fresh schema file per test.
    db_file = Path(tempfile.mkdtemp()) / "t.db"
    os.environ["CIDR_DB_PATH"] = str(db_file)
    init_db(connect())
    with TestClient(app) as c:
        yield c


def _create_pool(client, name="p", cidr="10.0.0.0/24"):
    r = client.post("/pools", json={"name": name, "cidr": cidr})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ---------- validation ----------

def test_reject_non_canonical_and_garbage(client):
    r = client.post("/pools", json={"name": "p", "cidr": "10.0.0.5/24"})
    assert r.status_code == 422
    r = client.post("/pools", json={"name": "p", "cidr": "not-a-cidr"})
    assert r.status_code == 422


def test_overlapping_pool_rejected(client):
    _create_pool(client, "a", "10.0.0.0/16")
    r = client.post("/pools", json={"name": "b", "cidr": "10.0.128.0/17"})
    assert r.status_code == 409
    # Same family, disjoint is fine.
    assert client.post("/pools", json={"name": "c", "cidr": "192.168.0.0/16"}).status_code == 201
    # IPv4 vs IPv6 never overlap.
    assert client.post("/pools", json={"name": "d", "cidr": "2001:db8::/32"}).status_code == 201


def test_reservation_bounds_and_family(client):
    pid = _create_pool(client, cidr="10.0.0.0/24")
    # out of bounds
    r = client.post(f"/pools/{pid}/reservations", json={"cidr": "10.0.1.0/24"})
    assert r.status_code == 422
    # wrong family
    r = client.post(f"/pools/{pid}/reservations", json={"cidr": "2001:db8::/64"})
    assert r.status_code == 422
    # non-canonical (host bits set)
    r = client.post(f"/pools/{pid}/reservations", json={"cidr": "10.0.0.5/26"})
    assert r.status_code == 422
    assert client.post(f"/pools/{pid}/reservations", json={"cidr": "10.0.0.0/26"}).status_code == 201
    # overlap with another reservation
    r = client.post(f"/pools/{pid}/reservations", json={"cidr": "10.0.0.0/28"})
    assert r.status_code == 409


def test_prefix_len_validation(client):
    pid = _create_pool(client, cidr="10.0.0.0/24")
    assert client.post(f"/pools/{pid}/leases", json={"prefix_len": 23}).status_code == 422
    assert client.post(f"/pools/{pid}/leases", json={"prefix_len": 33}).status_code == 422
    # v6 pool, v4-ish length fine, but prefix beyond 128 rejected
    v6 = _create_pool(client, "v6", "2001:db8::/32")
    assert client.post(f"/pools/{v6}/leases", json={"prefix_len": 129}).status_code == 422


def test_unknown_pool_and_token(client):
    assert client.get("/pools/999").status_code == 404
    assert client.post("/pools/999/leases", json={"prefix_len": 24}).status_code == 404
    assert client.get("/leases/nope").status_code == 404
    assert client.post("/leases/nope/renew", json={}).status_code == 404


# ---------- allocation semantics ----------

def test_smallest_start_allocation_and_free_view(client):
    pid = _create_pool(client, cidr="10.0.0.0/24")
    client.post(f"/pools/{pid}/reservations", json={"cidr": "10.0.0.0/26"})

    r = client.post(f"/pools/{pid}/leases", json={"prefix_len": 27})
    assert r.status_code == 201
    assert r.json()["cidr"] == "10.0.0.64/27"  # smallest legal start

    ov = client.get(f"/pools/{pid}").json()
    assert ov["free_cidrs"] == ["10.0.0.96/27", "10.0.0.128/25"]

    # next /27 starts at .96 (not .128)
    r = client.post(f"/pools/{pid}/leases", json={"prefix_len": 27})
    assert r.json()["cidr"] == "10.0.0.96/27"

    ov = client.get(f"/pools/{pid}").json()
    assert ov["free_cidrs"] == ["10.0.0.128/25"]


def test_exhaustion_409(client):
    pid = _create_pool(client, cidr="10.0.0.0/30")
    assert client.post(f"/pools/{pid}/leases", json={"prefix_len": 31}).status_code == 201
    assert client.post(f"/pools/{pid}/leases", json={"prefix_len": 31}).status_code == 201
    r = client.post(f"/pools/{pid}/leases", json={"prefix_len": 31})
    assert r.status_code == 409


def test_release_reclaims_space_and_old_token_conflicts(client):
    pid = _create_pool(client, cidr="10.0.0.0/30")
    token = client.post(f"/pools/{pid}/leases", json={"prefix_len": 31}).json()["token"]
    assert client.post(f"/leases/{token}/release").status_code == 200
    # space reclaimed: a new /31 fits again
    assert client.post(f"/pools/{pid}/leases", json={"prefix_len": 31}).status_code == 201
    # old token operations now conflict
    assert client.post(f"/leases/{token}/renew", json={}).status_code == 409
    assert client.post(f"/leases/{token}/release").status_code == 409
    assert client.get(f"/leases/{token}").json()["status"] == "released"


def test_renew_extends_expiry(client):
    pid = _create_pool(client, cidr="10.0.0.0/24")
    out = client.post(
        f"/pools/{pid}/leases", json={"prefix_len": 32, "ttl_seconds": 3600}
    ).json()
    r = client.post(f"/leases/{out['token']}/renew", json={"ttl_seconds": 7200})
    assert r.status_code == 200
    assert r.json()["expires_at"] == out["expires_at"] + 3600


def test_expiry_recycles_and_invalidates_token(client):
    pid = _create_pool(client, cidr="10.0.0.0/30")
    token = client.post(
        f"/pools/{pid}/leases", json={"prefix_len": 31, "ttl_seconds": 1}
    ).json()["token"]
    time.sleep(1.2)

    # expired token conflicts
    assert client.post(f"/leases/{token}/renew", json={}).status_code == 409
    assert client.post(f"/leases/{token}/release").status_code == 409
    assert client.get(f"/leases/{token}").json()["status"] == "expired"

    # recycled space can be leased again, yielding the same block
    r = client.post(f"/pools/{pid}/leases", json={"prefix_len": 31})
    assert r.status_code == 201
    assert r.json()["cidr"] == "10.0.0.0/31"

    # expired lease is gone from the active list
    active = client.get(f"/pools/{pid}").json()["active_leases"]
    assert all(l["token"] != token for l in active)


def test_ipv6_allocation(client):
    pid = _create_pool(client, "v6", "2001:db8::/32")
    client.post(f"/pools/{pid}/reservations", json={"cidr": "2001:db8::/48"})
    r = client.post(f"/pools/{pid}/leases", json={"prefix_len": 64})
    assert r.status_code == 201
    assert r.json()["cidr"] == "2001:db8:1::/64"
    ov = client.get(f"/pools/{pid}").json()
    assert ov["free_cidrs"][0] == "2001:db8:1:1::/64"
    assert ov["free_cidrs"][-1] == "2001:db8:8000::/33"


def test_concurrent_allocations_never_overlap(client):
    # /29 contains exactly eight /32 addresses.
    pid = _create_pool(client, cidr="192.0.2.0/29")

    def allocate(_):
        return client.post(f"/pools/{pid}/leases", json={"prefix_len": 32})

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(allocate, range(12)))

    successes = [r for r in results if r.status_code == 201]
    failures = [r for r in results if r.status_code == 409]
    assert len(successes) == 8
    assert len(failures) == 4
    cidrs = {r.json()["cidr"] for r in successes}
    assert cidrs == {f"192.0.2.{i}/32" for i in range(8)}
    tokens = [r.json()["token"] for r in successes]
    assert len(set(tokens)) == 8


# ---------- resize (in-place shrink / grow) ----------

def test_resize_shrink_keeps_low_address_subnet_and_frees_space(client):
    pid = _create_pool(client, cidr="10.0.0.0/24")
    out = client.post(f"/pools/{pid}/leases", json={"prefix_len": 24}).json()
    token = out["token"]

    r = client.put(f"/leases/{token}/resize", json={"target_prefix_len": 26})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cidr"] == "10.0.0.0/26"
    # token and expiry carry over unchanged
    assert body["token"] == token
    assert body["expires_at"] == out["expires_at"]
    assert body["renewed_at"] == out["renewed_at"]
    assert body["status"] == "active"

    ov = client.get(f"/pools/{pid}").json()
    assert [l["cidr"] for l in ov["active_leases"]] == ["10.0.0.0/26"]
    assert ov["free_cidrs"] == ["10.0.0.64/26", "10.0.0.128/25"]

    # freed space is allocatable again, starting at the smallest free address
    nxt = client.post(f"/pools/{pid}/leases", json={"prefix_len": 26})
    assert nxt.status_code == 201
    assert nxt.json()["cidr"] == "10.0.0.64/26"


def test_resize_grow_uses_unique_enclosing_supernet_and_shrink_back(client):
    pid = _create_pool(client, cidr="10.0.0.0/24")
    out = client.post(f"/pools/{pid}/leases", json={"prefix_len": 27}).json()
    token = out["token"]
    assert out["cidr"] == "10.0.0.0/27"

    r = client.put(f"/leases/{token}/resize", json={"target_prefix_len": 25})
    assert r.status_code == 200, r.text
    assert r.json()["cidr"] == "10.0.0.0/25"
    assert r.json()["token"] == token
    assert r.json()["expires_at"] == out["expires_at"]

    # shrinking again keeps the low-address subnet containing the start
    r = client.put(f"/leases/{token}/resize", json={"target_prefix_len": 26})
    assert r.status_code == 200, r.text
    assert r.json()["cidr"] == "10.0.0.0/26"

    # the freed part of the /25 can be re-leased
    nxt = client.post(f"/pools/{pid}/leases", json={"prefix_len": 26})
    assert nxt.status_code == 201
    assert nxt.json()["cidr"] == "10.0.0.64/26"


def test_resize_grow_blocked_by_reservation(client):
    pid = _create_pool(client, cidr="10.0.0.0/24")
    out = client.post(f"/pools/{pid}/leases", json={"prefix_len": 26}).json()
    # reservation in the sibling half of the enclosing /25
    assert (
        client.post(
            f"/pools/{pid}/reservations", json={"cidr": "10.0.0.64/26"}
        ).status_code
        == 201
    )

    r = client.put(f"/leases/{out['token']}/resize", json={"target_prefix_len": 25})
    assert r.status_code == 409
    # lease untouched
    assert client.get(f"/leases/{out['token']}").json()["cidr"] == "10.0.0.0/26"


def test_resize_grow_blocked_by_other_active_lease(client):
    pid = _create_pool(client, cidr="10.0.0.0/24")
    out = client.post(f"/pools/{pid}/leases", json={"prefix_len": 26}).json()
    other = client.post(f"/pools/{pid}/leases", json={"prefix_len": 26}).json()
    assert other["cidr"] == "10.0.0.64/26"

    r = client.put(f"/leases/{out['token']}/resize", json={"target_prefix_len": 25})
    assert r.status_code == 409
    assert client.get(f"/leases/{out['token']}").json()["cidr"] == "10.0.0.0/26"

    # releasing the neighbour unblocks the grow
    assert client.post(f"/leases/{other['token']}/release").status_code == 200
    r = client.put(f"/leases/{out['token']}/resize", json={"target_prefix_len": 25})
    assert r.status_code == 200
    assert r.json()["cidr"] == "10.0.0.0/25"


def test_resize_grow_outside_pool_conflicts(client):
    pid = _create_pool(client, cidr="10.0.0.64/26")
    out = client.post(f"/pools/{pid}/leases", json={"prefix_len": 27}).json()
    assert out["cidr"] == "10.0.0.64/27"
    # enclosing /26 is exactly the pool; enclosing /25 crosses the pool edge
    assert (
        client.put(
            f"/leases/{out['token']}/resize", json={"target_prefix_len": 26}
        ).status_code
        == 200
    )
    r = client.put(f"/leases/{out['token']}/resize", json={"target_prefix_len": 25})
    assert r.status_code == 409
    assert client.get(f"/leases/{out['token']}").json()["cidr"] == "10.0.0.64/26"


def test_resize_bad_prefix_422(client):
    pid = _create_pool(client, cidr="10.0.0.0/24")
    out = client.post(f"/pools/{pid}/leases", json={"prefix_len": 26}).json()
    assert (
        client.put(
            f"/leases/{out['token']}/resize", json={"target_prefix_len": 33}
        ).status_code
        == 422
    )
    assert (
        client.put(
            f"/leases/{out['token']}/resize", json={"target_prefix_len": -1}
        ).status_code
        == 422
    )
    # body / type errors rejected by the schema layer
    assert client.put(f"/leases/{out['token']}/resize", json={}).status_code == 422
    assert (
        client.put(
            f"/leases/{out['token']}/resize", json={"target_prefix_len": 1.5}
        ).status_code
        == 422
    )
    # the lease was never modified
    assert client.get(f"/leases/{out['token']}").json()["cidr"] == "10.0.0.0/26"


def test_resize_v6(client):
    pid = _create_pool(client, "v6", "2001:db8::/32")
    out = client.post(f"/pools/{pid}/leases", json={"prefix_len": 64}).json()
    assert out["cidr"] == "2001:db8::/64"

    r = client.put(f"/leases/{out['token']}/resize", json={"target_prefix_len": 65})
    assert r.status_code == 200
    assert r.json()["cidr"] == "2001:db8::/65"

    assert (
        client.put(
            f"/leases/{out['token']}/resize", json={"target_prefix_len": 129}
        ).status_code
        == 422
    )
    r = client.put(f"/leases/{out['token']}/resize", json={"target_prefix_len": 63})
    assert r.status_code == 200
    assert r.json()["cidr"] == "2001:db8::/63"


def test_resize_same_prefix_is_noop_without_event(client):
    pid = _create_pool(client, cidr="10.0.0.0/24")
    out = client.post(f"/pools/{pid}/leases", json={"prefix_len": 26}).json()

    r = client.put(f"/leases/{out['token']}/resize", json={"target_prefix_len": 26})
    assert r.status_code == 200
    assert r.json() == out

    events = client.get("/audit/events").json()
    assert [e["action"] for e in events] == ["allocate"]


def test_resize_requires_active_lease(client):
    pid = _create_pool(client, cidr="10.0.0.0/24")
    out = client.post(
        f"/pools/{pid}/leases", json={"prefix_len": 26, "ttl_seconds": 1}
    ).json()
    token = out["token"]

    time.sleep(1.2)
    assert (
        client.put(f"/leases/{token}/resize", json={"target_prefix_len": 25}).status_code
        == 409
    )

    # released lease cannot be resized either
    fresh = client.post(f"/pools/{pid}/leases", json={"prefix_len": 26}).json()
    client.post(f"/leases/{fresh['token']}/release")
    assert (
        client.put(
            f"/leases/{fresh['token']}/resize", json={"target_prefix_len": 25}
        ).status_code
        == 409
    )
    # unknown token
    assert (
        client.put("/leases/nope/resize", json={"target_prefix_len": 25}).status_code
        == 404
    )


def test_resize_audit_event(client):
    pid = _create_pool(client, cidr="10.0.0.0/24")
    lease = client.post(f"/pools/{pid}/leases", json={"prefix_len": 27}).json()
    assert (
        client.put(
            f"/leases/{lease['token']}/resize", json={"target_prefix_len": 25}
        ).status_code
        == 200
    )
    # a failed resize must not add an event: /23 would leave the /24 pool
    assert (
        client.put(
            f"/leases/{lease['token']}/resize", json={"target_prefix_len": 23}
        ).status_code
        == 409
    )
    assert (
        client.put(
            f"/leases/{lease['token']}/resize", json={"target_prefix_len": 24}
        ).status_code
        == 200
    )

    events = client.get("/audit/events").json()
    assert [e["action"] for e in events] == ["allocate", "resize", "resize"]
    first = events[1]
    assert first["cidr"] == "10.0.0.0/25"
    assert first["details"] == {
        "before_cidr": "10.0.0.0/27",
        "after_cidr": "10.0.0.0/25",
    }
    assert events[2]["cidr"] == "10.0.0.0/24"
    assert events[2]["details"] == {
        "before_cidr": "10.0.0.0/25",
        "after_cidr": "10.0.0.0/24",
    }
    assert first["lease_id"] == lease["id"]
    assert first["pool_id"] == pid
    # token never leaks into the audit log
    assert all(lease["token"] not in str(e) for e in events)
    _assert_chain_linkage(events)
    assert client.get("/audit/verify").json() == {"valid": True, "invalid_seq": None}


def test_concurrent_grow_and_allocate_never_overlap(client):
    # One /27 lease plus a free /27 in the same /26: a grow of the lease and a
    # fresh /26 allocation race for the same space; exactly one may win.
    pid = _create_pool(client, cidr="10.0.0.0/26")
    lease = client.post(f"/pools/{pid}/leases", json={"prefix_len": 27}).json()

    def grow(_):
        return client.put(
            f"/leases/{lease['token']}/resize", json={"target_prefix_len": 26}
        )

    def allocate(_):
        return client.post(f"/pools/{pid}/leases", json={"prefix_len": 26})

    with ThreadPoolExecutor(max_workers=12) as pool:
        grows = list(pool.map(grow, range(6)))
        allocs = list(pool.map(allocate, range(6)))

    grow_won = any(r.status_code == 200 for r in grows)
    alloc_won = any(r.status_code == 200 for r in allocs)
    assert grow_won + alloc_won == 1  # exactly one winner, never both

    ov = client.get(f"/pools/{pid}").json()
    cidrs = sorted(l["cidr"] for l in ov["active_leases"])
    assert cidrs in (["10.0.0.0/26"], ["10.0.0.0/27", "10.0.0.32/26"])


# ---------- audit chain ----------

GENESIS = "0" * 64


def _recompute(event):
    import hashlib
    import json

    fields = {
        k: event[k]
        for k in ("seq", "action", "lease_id", "pool_id", "cidr", "occurred_at", "details")
    }
    payload = event["prev_hash"] + json.dumps(
        fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assert_chain_linkage(events):
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    prev = GENESIS
    for e in events:
        assert e["prev_hash"] == prev
        assert e["hash"] == _recompute(e)
        prev = e["hash"]


def test_audit_allocate_renew_release_chain(client):
    pid = _create_pool(client, cidr="10.0.0.0/24")
    lease = client.post(
        f"/pools/{pid}/leases", json={"prefix_len": 27, "ttl_seconds": 3600}
    ).json()
    client.post(f"/leases/{lease['token']}/renew", json={"ttl_seconds": 7200})
    client.post(f"/leases/{lease['token']}/release")

    r = client.get("/audit/events")
    assert r.status_code == 200
    events = r.json()
    assert [e["action"] for e in events] == ["allocate", "renew", "release"]
    _assert_chain_linkage(events)

    alloc = events[0]
    assert alloc["lease_id"] == lease["id"]
    assert alloc["pool_id"] == pid
    assert alloc["cidr"] == lease["cidr"]
    assert alloc["details"]["prefix_len"] == 27
    assert alloc["details"]["ttl_seconds"] == 3600
    assert alloc["details"]["expires_at"] == lease["expires_at"]
    assert isinstance(alloc["occurred_at"], int)
    assert events[1]["details"]["ttl_seconds"] == 7200
    # the lease token (a bearer secret) never appears in the log
    assert all(lease["token"] not in str(e) for e in events)

    r = client.get("/audit/verify")
    assert r.status_code == 200
    assert r.json() == {"valid": True, "invalid_seq": None}


def test_audit_expire_event_recorded_on_sweep(client):
    pid = _create_pool(client, cidr="10.0.0.0/30")
    first = client.post(
        f"/pools/{pid}/leases", json={"prefix_len": 31, "ttl_seconds": 1}
    ).json()
    time.sleep(1.2)
    # next allocation sweeps the expired lease inside the same transaction
    assert client.post(f"/pools/{pid}/leases", json={"prefix_len": 31}).status_code == 201

    events = client.get("/audit/events").json()
    assert [e["action"] for e in events] == ["allocate", "expire", "allocate"]
    expire = events[1]
    assert expire["lease_id"] == first["id"]
    assert expire["cidr"] == first["cidr"]
    assert expire["details"]["expires_at"] == first["expires_at"]
    _assert_chain_linkage(events)
    assert client.get("/audit/verify").json() == {"valid": True, "invalid_seq": None}


def test_audit_failures_not_recorded(client):
    pid = _create_pool(client, cidr="10.0.0.0/30")
    assert client.post(f"/pools/{pid}/leases", json={"prefix_len": 31}).status_code == 201
    assert client.post(f"/pools/{pid}/leases", json={"prefix_len": 31}).status_code == 201

    # 409: pool exhausted; 404: unknown token; 422: bad prefix — none recorded
    assert client.post(f"/pools/{pid}/leases", json={"prefix_len": 31}).status_code == 409
    assert client.post("/leases/nope/renew", json={}).status_code == 404
    assert client.post(f"/pools/{pid}/leases", json={"prefix_len": 33}).status_code == 422

    events = client.get("/audit/events").json()
    assert [e["action"] for e in events] == ["allocate", "allocate"]
    assert client.get("/audit/verify").json() == {"valid": True, "invalid_seq": None}


def test_audit_verify_detects_tampering(client):
    pid = _create_pool(client, cidr="10.0.0.0/24")
    client.post(f"/pools/{pid}/leases", json={"prefix_len": 27})
    client.post(f"/pools/{pid}/leases", json={"prefix_len": 27})
    assert client.get("/audit/verify").json() == {"valid": True, "invalid_seq": None}

    # Tamper with the first event directly in the database.
    conn = connect()
    conn.execute("UPDATE audit_events SET cidr = '10.0.0.0/24' WHERE seq = 1")
    conn.close()

    assert client.get("/audit/verify").json() == {"valid": False, "invalid_seq": 1}


def test_audit_verify_detects_broken_linkage(client):
    pid = _create_pool(client, cidr="10.0.0.0/24")
    client.post(f"/pools/{pid}/leases", json={"prefix_len": 27})
    client.post(f"/pools/{pid}/leases", json={"prefix_len": 27})

    # Break the prev_hash link of the second event.
    conn = connect()
    conn.execute(f"UPDATE audit_events SET prev_hash = '{GENESIS}' WHERE seq = 2")
    conn.close()

    assert client.get("/audit/verify").json() == {"valid": False, "invalid_seq": 2}


def test_audit_empty_chain_is_valid(client):
    assert client.get("/audit/events").json() == []
    assert client.get("/audit/verify").json() == {"valid": True, "invalid_seq": None}

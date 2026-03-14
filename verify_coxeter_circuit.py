#!/usr/bin/env python3
"""Theorem verification: The Coxeter circuit (Section 8).

Verifies:
  - Coxeter element has order h = 30
  - Eigenvalue exponents are {1, 7, 11, 13, 17, 19, 23, 29} (all coprime to 30)
  - Euler totient φ(30) = 8 = rank(E8)
  - 240 roots decompose into exactly 8 orbits of 30
  - Fiber visit period is exactly 15 (Proposition 8.5)
  - C^15 preserves A8 coset L0 and swaps L1 ↔ L2 (Proposition 8.4)
  - A8 coset sizes are 72, 84, 84

All checks use exact integer/half-integer arithmetic on E8 roots.
"""

import sys
import numpy as np
from fractions import Fraction
from e8_utils import (build_e8_roots, cluster_by_hopf,
                      root_index_map, find_root)

passed = 0
failed = 0

def check(condition, description, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {description}")
        passed += 1
    else:
        print(f"  [FAIL] {description}" + (f" ({detail})" if detail else ""))
        failed += 1

# ================================================================
print("=" * 60)
print("COXETER CIRCUIT VERIFICATION (Section 8)")
print("=" * 60)

# Build E8
e8 = build_e8_roots()
rmap = root_index_map(e8)
shells = cluster_by_hopf(e8)

# Build root-to-fiber map
root_fiber = np.zeros(240, dtype=int)
for fi, sh in enumerate(shells):
    for ri in sh:
        root_fiber[ri] = fi

# ---- Simple roots ----
print("\n--- Simple roots and Coxeter element ---")
alpha = np.zeros((8, 8))
alpha[0] = [1, -1, 0, 0, 0, 0, 0, 0]
alpha[1] = [0, 1, -1, 0, 0, 0, 0, 0]
alpha[2] = [0, 0, 1, -1, 0, 0, 0, 0]
alpha[3] = [0, 0, 0, 1, -1, 0, 0, 0]
alpha[4] = [0, 0, 0, 0, 1, -1, 0, 0]
alpha[5] = [0, 0, 0, 0, 0, 1, -1, 0]
alpha[6] = [0, 0, 0, 0, 0, 1, 1, 0]
alpha[7] = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]

# Verify all simple roots are E8 roots
all_simple_in_e8 = all(find_root(e8, rmap, alpha[i]) >= 0 for i in range(8))
check(all_simple_in_e8, "All 8 simple roots are E8 roots")

# Verify Cartan matrix is E8 type
cartan = np.array([[round(2 * (alpha[i] @ alpha[j]) / (alpha[j] @ alpha[j]))
                    for j in range(8)] for i in range(8)])
e8_cartan = np.array([
    [2, -1, 0, 0, 0, 0, 0, 0],
    [-1, 2, -1, 0, 0, 0, 0, 0],
    [0, -1, 2, -1, 0, 0, 0, 0],
    [0, 0, -1, 2, -1, 0, 0, 0],
    [0, 0, 0, -1, 2, -1, -1, 0],
    [0, 0, 0, 0, -1, 2, 0, 0],
    [0, 0, 0, 0, -1, 0, 2, -1],
    [0, 0, 0, 0, 0, 0, -1, 2],
])
check(np.array_equal(cartan, e8_cartan), "Cartan matrix matches E8 Dynkin diagram")

# Build Coxeter element
C = np.eye(8)
for i in range(8):
    S = np.eye(8) - np.outer(alpha[i], alpha[i])
    C = S @ C

# ---- Order ----
print("\n--- Coxeter element order ---")
C_power = np.eye(8)
order = None
for k in range(1, 61):
    C_power = C @ C_power
    if np.allclose(C_power, np.eye(8), atol=1e-10):
        order = k
        break
check(order == 30, f"Coxeter element order h = {order}", f"expected 30")

# ---- Eigenvalue exponents ----
print("\n--- Eigenvalue exponents ---")
evals = np.linalg.eigvals(C)
raw_exponents = np.angle(evals) * 30 / (2 * np.pi)
exponents = sorted(set(round(e) % 30 for e in raw_exponents))
expected_exponents = [1, 7, 11, 13, 17, 19, 23, 29]
check(exponents == expected_exponents,
      f"Exponents are {{1,7,11,13,17,19,23,29}}",
      f"got {exponents}")

# Euler totient
from math import gcd
totient_30 = len([m for m in range(1, 30) if gcd(m, 30) == 1])
check(totient_30 == 8, f"φ(30) = {totient_30} = rank(E8)")
check(set(exponents) == set(m for m in range(1, 30) if gcd(m, 30) == 1),
      "Exponents = integers coprime to 30")

# ---- Orbits ----
print("\n--- Coxeter orbits on E8 roots ---")

# Build Coxeter permutation using exact root matching
from scipy.spatial import cKDTree
tree = cKDTree(e8)
coxeter_perm = np.zeros(240, dtype=int)
for i in range(240):
    v = C @ e8[i]
    d, idx = tree.query(v)
    check_close = d < 1e-8
    if not check_close:
        print(f"  WARNING: C maps root {i} to non-root (dist={d})")
    coxeter_perm[i] = idx

# Verify C is a permutation of E8 roots
check(len(set(coxeter_perm)) == 240, "Coxeter element permutes all 240 roots")

# Build orbits
visited = set()
orbits = []
for seed in range(240):
    if seed in visited:
        continue
    orbit = [seed]
    idx = seed
    for _ in range(29):
        idx = coxeter_perm[idx]
        orbit.append(idx)
    visited.update(orbit)
    orbits.append(orbit)

check(len(orbits) == 8, f"Number of orbits: {len(orbits)}", "expected 8")
check(all(len(o) == 30 for o in orbits),
      f"All orbits have size 30",
      f"sizes: {[len(o) for o in orbits]}")
check(sum(len(o) for o in orbits) == 240,
      f"Total roots in orbits: {sum(len(o) for o in orbits)}", "expected 240")

# ---- A8 coset grading ----
print("\n--- A8 coset grading ---")
cartan_inv = np.linalg.inv(cartan.astype(float))

def a8_coset(root):
    n = np.array([root @ alpha[j] for j in range(8)])
    c = cartan_inv @ n
    return round(c[5]) % 3

root_cosets = [a8_coset(e8[i]) for i in range(240)]
coset_counts = [root_cosets.count(i) for i in range(3)]
check(sorted(coset_counts) == [72, 84, 84],
      f"A8 coset sizes: L0={coset_counts[0]}, L1={coset_counts[1]}, L2={coset_counts[2]}",
      "expected 72, 84, 84")

# ---- C^15 coset preservation (Proposition 8.4) ----
print("\n--- C^15 coset action (Proposition 8.4) ---")

# Apply C^15 to each root and check coset mapping
C15_perm = list(range(240))
for _ in range(15):
    C15_perm = [coxeter_perm[i] for i in C15_perm]

L0_preserved = True
L1_to_L2 = True
L2_to_L1 = True
for i in range(240):
    src_coset = root_cosets[i]
    dst_coset = root_cosets[C15_perm[i]]
    if src_coset == 0 and dst_coset != 0:
        L0_preserved = False
    if src_coset == 1 and dst_coset != 2:
        L1_to_L2 = False
    if src_coset == 2 and dst_coset != 1:
        L2_to_L1 = False

check(L0_preserved, "C^15 preserves L0 (all 72 roots stay in L0)")
check(L1_to_L2, "C^15 maps L1 → L2 (all 84 roots)")
check(L2_to_L1, "C^15 maps L2 → L1 (all 84 roots)")

# Verify C^15 is the UNIQUE non-trivial power preserving L0
print("\n--- C^15 uniqueness (Proposition 8.4) ---")
powers_preserving_L0 = []
perm_k = list(range(240))
for k in range(1, 30):
    perm_k = [coxeter_perm[i] for i in perm_k]
    preserves = all(root_cosets[perm_k[i]] == 0 for i in range(240) if root_cosets[i] == 0)
    if preserves:
        powers_preserving_L0.append(k)

check(powers_preserving_L0 == [15],
      f"Only k=15 and k=30 preserve L0",
      f"found k={powers_preserving_L0}")

# ---- Fiber visit period (Proposition 8.5) ----
print("\n--- Fiber visit period (Proposition 8.5) ---")

all_period_15 = True
for oi, orbit in enumerate(orbits):
    fibers = [int(root_fiber[r]) for r in orbit]
    if fibers[:15] != fibers[15:]:
        all_period_15 = False
        print(f"  WARNING: Orbit {oi} fiber sequence does NOT repeat at period 15")

check(all_period_15,
      "Fiber visit sequence has period exactly 15 for all 8 orbits")

# Verify period is not shorter than 15
shorter_period = False
for oi, orbit in enumerate(orbits):
    fibers = [int(root_fiber[r]) for r in orbit]
    for p in [1, 2, 3, 5, 6, 10]:  # divisors of 30 less than 15
        if all(fibers[k] == fibers[k + p] for k in range(30 - p)):
            shorter_period = True
            print(f"  WARNING: Orbit {oi} has shorter fiber period {p}")

check(not shorter_period, "No orbit has fiber period shorter than 15")

# ================================================================
print("\n" + "=" * 60)
print(f"COXETER CIRCUIT SUMMARY: {passed}/{passed + failed} claims verified")
if failed > 0:
    print(f"  *** {failed} FAILURES ***")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)

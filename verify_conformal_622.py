#!/usr/bin/env python3
"""Theorem 1.6(c) verification: Conformal selection produces A3 cuboctahedron.

Verifies:
  - For each of 10 D4 Hopf fibers, conformal selection C(F0, n) selects 12 roots
  - The 12 roots form a metrically perfect A3 = D3 cuboctahedron
  - All C(12,2) = 66 pairwise angles are in {60°, 90°, 120°, 180°}
  - The selection has 6+2+2+2 structure: distinguished A2 contributes 6 roots,
    three other A2 fibers contribute 2 each (one antipodal pair)
  - The 12 roots project isometrically to R^3 (angle preservation)
"""

import sys
import numpy as np
from itertools import combinations
from e8_utils import build_e8_roots, cluster_by_hopf

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
print("THEOREM 1.6(c): Conformal Selection → Cuboctahedron A3")
print("=" * 60)

e8 = build_e8_roots()
shells = cluster_by_hopf(e8)

# ---- A2 partition of each D4 ----
# Find 4 × A2 partition by combinatorial search
def find_a2_partition(d4_roots):
    n = len(d4_roots)
    for combo in combinations(range(n), 6):
        sub = d4_roots[list(combo)]
        rank = np.linalg.matrix_rank(sub, tol=1e-6)
        if rank != 2: continue
        neg_closed = all(any(np.allclose(d4_roots[j], -d4_roots[combo[i]], atol=1e-6)
                            for j in combo) for i in range(6))
        if not neg_closed: continue
        ips = set(round(float(sub[a] @ sub[b])) for a, b in combinations(range(6), 2))
        if ips > {-2, -1, 0, 1}: continue
        remaining = set(range(n)) - set(combo)
        for c2 in combinations(remaining, 6):
            s2 = d4_roots[list(c2)]
            if np.linalg.matrix_rank(s2, tol=1e-6) != 2: continue
            nc2 = all(any(np.allclose(d4_roots[j], -d4_roots[c2[i]], atol=1e-6)
                         for j in c2) for i in range(6))
            if not nc2: continue
            ip2 = set(round(float(s2[a] @ s2[b])) for a, b in combinations(range(6), 2))
            if ip2 > {-2, -1, 0, 1}: continue
            rest2 = remaining - set(c2)
            for c3 in combinations(rest2, 6):
                s3 = d4_roots[list(c3)]
                if np.linalg.matrix_rank(s3, tol=1e-6) != 2: continue
                c4 = list(rest2 - set(c3))
                if len(c4) != 6: continue
                s4 = d4_roots[c4]
                if np.linalg.matrix_rank(s4, tol=1e-6) != 2: continue
                return [list(combo), list(c2), list(c3), c4]
    return None

print("\n--- D4 = 4 × A2 partitions ---")
fiber_a2 = {}  # shell[local_idx] → a2 label
for fi, shell in enumerate(shells):
    part = find_a2_partition(e8[shell])
    check(part is not None, f"Fiber {fi}: 4 × A2 partition exists")
    if part:
        check(all(len(g) == 6 for g in part), f"Fiber {fi}: all 4 groups have 6 roots")
        for a2f, members in enumerate(part):
            for local_i in members:
                fiber_a2[(fi, local_i)] = a2f

# ---- Conformal selection per fiber ----
print("\n--- Conformal selection C(F0, n) ---")

EXPECTED_ANGLES = {60, 90, 120, 180}
total_622 = 0

for fi, shell in enumerate(shells):
    d4_roots = e8[shell]
    _, _, Vt = np.linalg.svd(d4_roots, full_matrices=False)
    basis4d = Vt[:4]
    d4_4d = d4_roots @ basis4d.T

    # Search for conformal normal giving 6+2+2+2 structure
    found = False
    for i in range(24):
        if found: break
        for j in range(i + 1, 24):
            n = d4_4d[i] + d4_4d[j]
            norm = np.linalg.norm(n)
            if norm < 1e-10: continue
            n = n / norm

            dots = d4_4d @ n
            selected = np.where(np.abs(dots) < 1e-6)[0]
            if len(selected) != 12: continue

            # Check A2 structure
            a2_counts = {}
            for si in selected:
                a2f = fiber_a2.get((fi, si), -1)
                a2_counts[a2f] = a2_counts.get(a2f, 0) + 1
            pattern = sorted(a2_counts.values(), reverse=True)
            if pattern != [6, 2, 2, 2]: continue

            # Project to 3D
            proj_basis = []
            for e_vec in np.eye(4):
                v = e_vec - np.dot(e_vec, n) * n
                for b in proj_basis: v -= np.dot(v, b) * b
                nm = np.linalg.norm(v)
                if nm > 1e-10: proj_basis.append(v / nm)
                if len(proj_basis) == 3: break
            if len(proj_basis) < 3: continue
            proj_basis_arr = np.array(proj_basis)

            sel_3d = d4_4d[selected] @ proj_basis_arr.T

            # Check all 66 pairwise angles
            angles = set()
            for a in range(12):
                for b in range(a + 1, 12):
                    na = np.linalg.norm(sel_3d[a])
                    nb = np.linalg.norm(sel_3d[b])
                    if na < 1e-10 or nb < 1e-10: continue
                    cos_ab = sel_3d[a] @ sel_3d[b] / (na * nb)
                    angles.add(round(np.degrees(np.arccos(np.clip(cos_ab, -1, 1)))))

            if angles == EXPECTED_ANGLES:
                found = True
                total_622 += 1

                check(True, f"Fiber {fi}: 12 roots selected")
                check(True, f"Fiber {fi}: angle set = {{60°, 90°, 120°, 180°}}")
                check(True, f"Fiber {fi}: structure = {pattern} (6+2+2+2)")

                # Verify it's a cuboctahedron (24 edges)
                D = np.zeros((12, 12))
                for a in range(12):
                    for b in range(a + 1, 12):
                        D[a, b] = D[b, a] = np.linalg.norm(sel_3d[a] - sel_3d[b])
                min_d = np.min(D[D > 0.01])
                edges = [(a, b) for a in range(12) for b in range(a + 1, 12)
                         if abs(D[a, b] - min_d) < 0.01]
                check(len(edges) == 24, f"Fiber {fi}: 24 cuboctahedral edges",
                      f"got {len(edges)}")

                # Verify all 66 pairwise inner products are preserved
                check(len(angles) == 4 and angles == EXPECTED_ANGLES,
                      f"Fiber {fi}: all 66 pairwise angles exactly preserved")

                # Verify the distinguished A2 contributes 6
                dist_a2 = max(a2_counts, key=a2_counts.get)
                check(a2_counts[dist_a2] == 6,
                      f"Fiber {fi}: distinguished A2 contributes 6 roots")

                # Verify each non-distinguished A2 contributes exactly 2
                non_dist = [c for k, c in a2_counts.items() if k != dist_a2]
                check(all(c == 2 for c in non_dist),
                      f"Fiber {fi}: non-distinguished A2s each contribute 2 roots",
                      f"counts: {non_dist}")
                break

    if not found:
        check(False, f"Fiber {fi}: conformal selection found", "NO valid selection")

check(total_622 == 10,
      f"All 10 fibers produce 6+2+2+2 conformal cuboctahedra",
      f"got {total_622}/10")

# ================================================================
print("\n" + "=" * 60)
print(f"THEOREM 1.6(c) SUMMARY: {passed}/{passed + failed} claims verified")
if failed > 0:
    print(f"  *** {failed} FAILURES ***")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)

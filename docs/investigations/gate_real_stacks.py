"""Fuse the user's two real stacks with confidence-gated exponents.

The regression p4 could not pass is here, not in the synthetic control: stack 1's
background is defocused in every frame, and a fixed exponent commits to noise-driven
winners there (lab_drift 9.96 at p1 -> 22.51 at p4). Writes into weight_experiment/ so
original_stacks_p4.py can score these next to the fixed-exponent results.

    python docs/investigations/gate_real_stacks.py
"""
import json
import resource
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from laplacian_weight_experiment import GatedStacker

ROOT = Path('test_outputs/real_stacks')
STACKS = ['focus_stack_test-1', 'focus_stack_test-2']
VARIANTS = {
    'gate4_pct10':    dict(power=4, floor_pct=10, floor_scale=1.0, gate_blur=0),
    'gate4_pct10_b4': dict(power=4, floor_pct=10, floor_scale=1.0, gate_blur=4),
    'gate4_pct25_b4': dict(power=4, floor_pct=25, floor_scale=1.0, gate_blur=4),
    'gate8_pct10_b4': dict(power=8, floor_pct=10, floor_scale=1.0, gate_blur=4),
}


def main():
    cv2.setNumThreads(4)
    only = sys.argv[1:] or list(VARIANTS)
    for stack in STACKS:
        base = ROOT / stack
        paths = sorted((base / 'native_aligned').glob('*.png'))
        outdir = base / 'weight_experiment'
        outdir.mkdir(parents=True, exist_ok=True)
        for name in only:
            kw = VARIANTS[name]
            start = time.perf_counter()
            s = GatedStacker(**kw)
            res = s.stack(paths)
            elapsed = time.perf_counter() - start
            assert cv2.imwrite(str(outdir / f'{name}.png'), res)
            meta = dict(name=name, stack=stack, frames=len(paths), seconds=elapsed,
                        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                        gate_stats=s.gate_stats, **kw)
            (outdir / f'{name}.json').write_text(json.dumps(meta, indent=2))
            gated = [f"{g['gated_fraction']:.2f}" for g in s.gate_stats]
            print(f'{stack} {name}: {elapsed:.1f}s  gated_fraction/level=[{", ".join(gated)}]',
                  flush=True)


if __name__ == '__main__':
    main()

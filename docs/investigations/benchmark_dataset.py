"""Run both fusion algorithms over a fixed subset of the research focus-stack dataset.

The dataset ships pre-registered frames in each scene's `aligned/` directory, so
fusion runs with alignment disabled and both algorithms see byte-identical inputs.
That keeps this measurement about fusion only, unlike docs/PHASE6_TESTING.md.

Each scene also ships Helicon Focus renders. Those are a competitive reference for
side-by-side inspection, NOT ground truth: scoring against them would make
"reproduce Helicon, artifacts included" the objective. They are copied through for
viewing and diffing, and deliberately never enter a score.

    python docs/investigations/benchmark_dataset.py --width 2592
    python docs/investigations/benchmark_dataset.py --scenes olympus_macro_lens/needle2 --width 0

width=0 preserves native 5184x3888. Outputs are local evaluation artifacts.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

DATASET = Path('test_stacks/dataset/focus_stack_dataset/focus_stack_dataset/dataset')

# Frozen regression subset: three olympus macro (fine detail, shallow DOF) and three
# lumix (depth-layered scenes with smooth backgrounds that expose blending noise).
DEFAULT_SCENES = [
    'test/olympus_macro_lens/needle2',
    'test/olympus_macro_lens/chess_piece3',
    'test/olympus_macro_lens/flower3',
    'test/lumix_lens/leaf2',
    'test/lumix_lens/chairs3',
    'test/lumix_lens/eastern_red_cedar_tree2',
]
ALGORITHMS = ['laplacian', 'complex_wavelet']
FUSE = Path('docs/investigations/evaluate_real_stacks.py')
WEIGHTED = Path('docs/investigations/laplacian_weight_experiment.py')


def parse_weighted(name):
    """'laplacian_p2' / 'laplacian_p4_s1' / 'laplacian_p4_c1' -> (power, smoothing, coarse).

    These route to WeightedStacker in laplacian_weight_experiment.py rather than
    production FocusStacker, so experimental weighting is measured by the same
    harness as the shipped algorithms.
    """
    if not name.startswith('laplacian_p'):
        return None
    rest = name[len('laplacian_p'):]
    smoothing, coarse = 0.0, 0
    if '_s' in rest:
        rest, s = rest.split('_s', 1)
        smoothing = float(s)
    elif '_c' in rest:
        rest, c = rest.split('_c', 1)
        coarse = int(c)
    return float(rest), smoothing, coarse


def stage(scene_dir: Path, work: Path, width: int, step: int) -> Path:
    """Materialise the frames to fuse. Returns a directory of images.

    Native, every-frame runs point straight at `aligned/` with no copy.
    """
    aligned = scene_dir / 'aligned'
    frames = sorted(aligned.glob('*.jpg'), key=lambda p: int(p.stem.split('_')[1]))
    if not frames:
        raise SystemExit(f'No aligned frames in {aligned}')
    frames = frames[::step]
    if not width and step == 1:
        return aligned, len(frames)
    work.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(frames):
        im = cv2.imread(str(src))
        if im is None:
            raise SystemExit(f'Cannot read {src}')
        if width and im.shape[1] > width:
            im = cv2.resize(im, (width, round(im.shape[0] * width / im.shape[1])),
                            interpolation=cv2.INTER_AREA)
        assert cv2.imwrite(str(work / f'{i:02d}.png'), im)
    return work, len(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenes', nargs='*', default=None,
                    help='scene paths relative to the dataset root (default: frozen subset)')
    ap.add_argument('--width', type=int, default=2592,
                    help='resize width; 0 keeps native 5184 (default 2592, half)')
    ap.add_argument('--step', type=int, default=1,
                    help='use every Nth frame; 1 keeps all 30 (default 1)')
    ap.add_argument('--algorithms', nargs='*', default=ALGORITHMS)
    ap.add_argument('--out', type=Path, default=Path('test_outputs/dataset_benchmark'))
    ap.add_argument('--dataset', type=Path, default=DATASET)
    args = ap.parse_args()

    scenes = args.scenes if args.scenes is not None else DEFAULT_SCENES
    scenes = [s if s.startswith(('test/', 'train/')) else f'test/{s}' for s in scenes]
    args.out.mkdir(parents=True, exist_ok=True)
    records = []

    for scene in scenes:
        scene_dir = args.dataset / scene
        if not scene_dir.is_dir():
            raise SystemExit(f'Missing scene: {scene_dir}')
        tag = scene.replace('/', '__')
        dest = args.out / tag
        dest.mkdir(parents=True, exist_ok=True)
        work = dest / 'inputs'
        src_dir, n_frames = stage(scene_dir, work, args.width, args.step)

        # Carry the Helicon reference through for inspection, matched to run scale.
        helicon = scene_dir / 'helicon_focus.jpg'
        if helicon.exists():
            him = cv2.imread(str(helicon))
            if args.width and him is not None and him.shape[1] > args.width:
                him = cv2.resize(him, (args.width, round(him.shape[0] * args.width / him.shape[1])),
                                 interpolation=cv2.INTER_AREA)
            assert cv2.imwrite(str(dest / 'helicon_reference.png'), him)

        for algo in args.algorithms:
            print(f'[{tag}] {algo}: {n_frames} frames', flush=True)
            weighted = parse_weighted(algo)
            if weighted:
                power, smoothing, coarse = weighted
                cmd = [sys.executable, str(WEIGHTED), str(src_dir), str(dest),
                       '--power', repr(power), '--smoothing', repr(smoothing),
                       '--coarse-levels', str(coarse)]
            else:
                cmd = [sys.executable, str(FUSE), 'fuse', str(src_dir), str(dest),
                       '--algorithm', algo]
            started = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            wall = time.perf_counter() - started
            if proc.returncode != 0:
                print(f'  FAILED rc={proc.returncode}\n{proc.stderr[-2000:]}', flush=True)
                records.append(dict(scene=scene, algorithm=algo, frames=n_frames,
                                    width=args.width, step=args.step, failed=True,
                                    returncode=proc.returncode, stderr=proc.stderr[-2000:]))
                continue
            stats = json.loads(proc.stdout.strip().splitlines()[-1])
            if weighted:
                # normalise the weight script's 'pN_sM' artefact names and key spelling
                src_name = (f'p{weighted[0]:g}_c{weighted[2]}' if weighted[2]
                            else f'p{weighted[0]:g}_s{weighted[1]:g}')
                for ext in ('png', 'json'):
                    old = dest / f'{src_name}.{ext}'
                    if old.exists():
                        old.replace(dest / f'{algo}.{ext}')
                stats['fusion_seconds'] = stats.pop('seconds', wall)
                stats['algorithm'] = algo
            stats.update(scene=scene, width=args.width, step=args.step, wall_seconds=wall,
                         failed=False)
            print(f'  {stats["fusion_seconds"]:.1f}s fusion, '
                  f'{stats["peak_rss_mib"]/1024:.2f} GiB peak RSS', flush=True)
            records.append(stats)

        if work.exists():
            shutil.rmtree(work)   # staged copies are reproducible; results are not

    summary = args.out / f'benchmark_w{args.width}_s{args.step}.json'
    summary.write_text(json.dumps(records, indent=2))
    print(f'\nWrote {summary}')
    ok = [r for r in records if not r.get('failed')]
    if ok:
        print(f'{len(ok)}/{len(records)} runs succeeded; '
              f'peak RSS max {max(r["peak_rss_mib"] for r in ok)/1024:.2f} GiB')


if __name__ == '__main__':
    main()

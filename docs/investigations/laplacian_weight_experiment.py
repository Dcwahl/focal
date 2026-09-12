"""Experimental focus weighting; does not change application defaults.

python docs/investigations/laplacian_weight_experiment.py INPUT OUTPUT --power 4
INPUT must contain only one sequence, already aligned when needed.
"""
import argparse
import json
import resource
import time
from pathlib import Path

import cv2
import numpy as np

from focal.core.stacker import FocusStacker


class WeightedStacker(FocusStacker):
    def __init__(self, power=1, smoothing=0):
        super().__init__(skip_alignment=True)
        self.power = power
        self.smoothing = smoothing

    def _compute_weights(self, focus_measures, decisive=True):
        # `decisive` is production's per-level flag. Ignored here on purpose: this class
        # measures a uniform exponent applied at every level, base included, which is the
        # variant the recorded p2/p4/p8 numbers describe.
        if self.power == 1 and self.smoothing == 0:
            return super()._compute_weights(focus_measures)
        measures = np.stack(focus_measures)
        maximum = measures.max(axis=0, keepdims=True)
        np.divide(measures, maximum, out=measures, where=maximum > 0)
        np.power(measures, self.power, out=measures)
        total = measures.sum(axis=0, keepdims=True)
        weights = np.full_like(measures, 1 / len(measures))
        np.divide(measures, total, out=weights, where=total > 0)
        if self.smoothing:
            for i in range(len(weights)):
                weights[i] = cv2.GaussianBlur(weights[i], (0, 0), self.smoothing)
            weights /= weights.sum(axis=0, keepdims=True)
        return list(weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--power', type=float, default=1)
    parser.add_argument('--smoothing', type=float, default=0)
    parser.add_argument('--coarse-levels', type=int, default=0,
                        help='average (p=1) this many coarsest pyramid levels')
    args = parser.parse_args()
    if args.power < 1 or args.smoothing < 0:
        parser.error('power must be >= 1 and smoothing >= 0')
    if args.coarse_levels and args.smoothing:
        parser.error('--coarse-levels and --smoothing are separate experiments')
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(4)
    start = time.perf_counter()
    if args.coarse_levels:
        stacker = LevelGatedStacker(args.power, args.coarse_levels)
    else:
        stacker = WeightedStacker(args.power, args.smoothing)
    result = stacker.stack(paths)
    elapsed = time.perf_counter() - start
    name = (f'p{args.power:g}_c{args.coarse_levels}' if args.coarse_levels
            else f'p{args.power:g}_s{args.smoothing:g}')
    assert cv2.imwrite(str(args.output / f'{name}.png'), result)
    metrics = dict(power=args.power, smoothing=args.smoothing,
                   coarse_levels=args.coarse_levels, input=str(args.input.resolve()),
                   frames=len(paths), shape=list(result.shape), seconds=elapsed,
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (args.output / f'{name}.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)




class GatedStacker(FocusStacker):
    """Power weighting whose exponent is gated by how much focus evidence a pixel has.

    A fixed exponent is decisive everywhere, including the regions where every frame is
    defocused and the sharpest-frame ranking is just noise; committing there is what made
    p4 blotch the background of the real stacks. The exponent is therefore driven per
    pixel by the peak focus measure relative to a floor estimated from the image, so a
    pixel with no real focus signal falls back to p=1 (the shipped proportional blend)
    and only pixels with evidence get the decisive exponent.
    """

    def __init__(self, power=4, floor_pct=10.0, floor_scale=1.0, gate_blur=0.0):
        super().__init__(skip_alignment=True)
        self.power = power
        self.floor_pct = floor_pct
        self.floor_scale = floor_scale
        # "No frame is in focus here" is a property of a region, not of one pixel, and the
        # peak measure is noisy. Pooling it first stops the exponent from varying pixel to
        # pixel, which would reintroduce the blotchiness the gate exists to remove.
        self.gate_blur = gate_blur
        self.gate_stats = []

    def _compute_weights(self, focus_measures, decisive=True):
        measures = np.stack(focus_measures)
        m_max = measures.max(axis=0)
        if self.gate_blur:
            m_max = cv2.GaussianBlur(m_max, (0, 0), self.gate_blur)
        tau = self.floor_scale * float(np.percentile(m_max, self.floor_pct))
        if tau <= 0:
            return super()._compute_weights(focus_measures)
        confidence = m_max / (m_max + tau)
        exponent = 1.0 + (self.power - 1.0) * confidence
        norm = np.divide(measures, m_max, out=np.zeros_like(measures), where=m_max > 0)
        np.power(norm, exponent, out=norm)
        total = norm.sum(axis=0)
        weights = np.full_like(norm, 1.0 / len(focus_measures))
        np.divide(norm, total, out=weights, where=total > 0)
        self.gate_stats.append(dict(tau=tau, mean_exponent=float(exponent.mean()),
                                    gated_fraction=float((confidence < 0.5).mean())))
        return list(weights)


class LevelGatedStacker(FocusStacker):
    """Decisive at fine pyramid levels, proportional at coarse ones.

    The background blotching a fixed exponent causes on the real stacks is not noisy focus
    ranking - measured there, peak sharpness sits at the 43rd percentile and the winner is
    spatially coherent. It is inter-frame brightness disagreement: that region's mean luma
    swings 29.8..101.7 across the frames and its cross-frame variation is almost entirely
    low-frequency. A decisive exponent therefore picks between frames that differ in
    exposure, not in focus, and the choice shows up as coarse patches.

    The Laplacian pyramid already separates those bands, so the exponent can too: commit at
    the fine levels that carry detail, average at the coarse levels that carry brightness.
    `coarse_levels` counts from the coarsest end; those levels fall back to p=1.
    """

    def __init__(self, power=4, coarse_levels=1):
        super().__init__(skip_alignment=True)
        self.power = power
        self.coarse_levels = coarse_levels
        self._level = 0
        self.level_powers = []

    def stack(self, *a, **kw):
        self._level = 0
        self.level_powers = []
        return super().stack(*a, **kw)

    def _compute_weights(self, focus_measures, decisive=True):
        # Tracks its own level so `coarse_levels` > 1 stays testable; production only
        # ever exempts the single coarsest level, so its flag is not enough here.
        num_levels = self.num_levels or self._num_levels_seen
        level = self._level
        self._level += 1
        decisive = level < num_levels - self.coarse_levels
        power = self.power if decisive else 1
        self.level_powers.append(power)
        if power == 1:
            return super()._compute_weights(focus_measures)
        measures = np.stack(focus_measures)
        maximum = measures.max(axis=0, keepdims=True)
        np.divide(measures, maximum, out=measures, where=maximum > 0)
        np.power(measures, power, out=measures)
        total = measures.sum(axis=0, keepdims=True)
        weights = np.full_like(measures, 1 / len(measures))
        np.divide(measures, total, out=weights, where=total > 0)
        return list(weights)

    def _compute_num_levels(self, shape):
        self._num_levels_seen = super()._compute_num_levels(shape)
        return self._num_levels_seen


if __name__ == '__main__':
    main()


class ConfidenceGatedStacker(FocusStacker):
    """Level gate (inherited from production) plus a confidence gate on the fine levels.

    The two gates answer different questions and neither covers the other. Production's
    level gate keeps inter-frame *brightness* disagreement out of the exponent by leaving
    the base level proportional. This adds the gate for the other mechanism: in regions
    where no frame has real focus evidence, the sharpest-frame ranking is noise, and a
    decisive exponent commits to it as grain. The exponent is therefore pulled back
    toward 1 where the peak focus measure sits near a floor estimated from the image.

    `measure_blur` pools each frame's focus measure before ranking, on the grounds that
    some of the grain is the measure being noisy per pixel rather than the choice being
    wrong. `margin` scores confidence by how far the winner beats the runner-up instead
    of by absolute magnitude.
    """

    def __init__(self, focus_power=4.0, floor_pct=10.0, gate_blur=0.0,
                 measure_blur=0.0, margin=False):
        super().__init__(skip_alignment=True, focus_power=focus_power)
        self.floor_pct = floor_pct
        self.gate_blur = gate_blur
        self.measure_blur = measure_blur
        self.margin = margin
        self.gate_stats = []

    def stack(self, *a, **kw):
        self.gate_stats = []
        return super().stack(*a, **kw)

    def _compute_weights(self, focus_measures, decisive=True):
        # The base level stays proportional: that is production's level gate, and the
        # confidence gate has no business overriding it.
        if not decisive or self.focus_power == 1.0:
            return super()._compute_weights(focus_measures, decisive=decisive)

        if self.measure_blur:
            focus_measures = [cv2.GaussianBlur(m, (0, 0), self.measure_blur)
                              for m in focus_measures]
        measures = np.stack(focus_measures)
        m_max = measures.max(axis=0)

        if self.margin:
            # How far clear is the winner? Partition rather than sort: only the
            # runner-up matters and these arrays are large.
            second = np.partition(measures, -2, axis=0)[-2]
            confidence = np.divide(m_max - second, m_max,
                                   out=np.zeros_like(m_max), where=m_max > 0)
        else:
            pooled = cv2.GaussianBlur(m_max, (0, 0), self.gate_blur) if self.gate_blur else m_max
            tau = float(np.percentile(pooled, self.floor_pct))
            confidence = pooled / (pooled + tau) if tau > 0 else np.ones_like(pooled)

        exponent = 1.0 + (self.focus_power - 1.0) * confidence
        norm = np.divide(measures, m_max, out=np.zeros_like(measures), where=m_max > 0)
        np.power(norm, exponent, out=norm)
        total = norm.sum(axis=0)
        weights = np.full_like(norm, 1.0 / len(focus_measures))
        np.divide(norm, total, out=weights, where=total > 0)
        self.gate_stats.append(dict(mean_exponent=float(exponent.mean()),
                                    mean_confidence=float(confidence.mean())))
        return list(weights)


class PooledStacker(FocusStacker):
    """Production's exponent and level gate, with the focus measure pooled before ranking.

    The gate sweep showed the confidence floor barely moves the grain while pooling moves
    it a lot, in both the flat and the textured zones. That points at the focus measure
    itself being noisy per pixel rather than the selection being wrong, which is a much
    simpler thing to fix. This is the ablation: exponent + level gate + pooling, no
    confidence gate at all.
    """

    def __init__(self, focus_power=8.0, measure_blur=4.0):
        super().__init__(skip_alignment=True, focus_power=focus_power)
        self.measure_blur = measure_blur

    def _compute_weights(self, focus_measures, decisive=True):
        if self.measure_blur:
            focus_measures = [cv2.GaussianBlur(m, (0, 0), self.measure_blur)
                              for m in focus_measures]
        return super()._compute_weights(focus_measures, decisive=decisive)

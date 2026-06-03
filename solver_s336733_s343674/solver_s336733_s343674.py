import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .abstract_solver import AbstractSolver

EPS = 1e-7
ROUND_DIGITS = 6


@dataclass
class ItemData:
    id: str
    width: float
    depth: float
    height: float
    weight: float
    value: float
    allowed: str
    volume: float
    base_area: float
    max_dim: float


@dataclass
class VehicleData:
    type: str
    width: float
    depth: float
    height: float
    max_weight: float
    cost: float
    max_value: float
    gravity: float
    volume: float


@dataclass
class Box:
    id: str
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    w: float
    d: float
    h: float
    orient: int
    weight: float
    value: float

    @property
    def base_area(self) -> float:
        return self.w * self.d


@dataclass
class BinState:
    idx: int
    vehicle: VehicleData
    boxes: List[Box] = field(default_factory=list)
    candidate_points: set = field(default_factory=lambda: {(0.0, 0.0, 0.0)})
    used_weight: float = 0.0
    used_value: float = 0.0
    used_volume: float = 0.0

    def clone(self) -> "BinState":
        return BinState(
            idx=self.idx,
            vehicle=self.vehicle,
            boxes=list(self.boxes),
            candidate_points=set(self.candidate_points),
            used_weight=self.used_weight,
            used_value=self.used_value,
            used_volume=self.used_volume,
        )


class solver_s336733_s343674(AbstractSolver):
    ROTATIONS = (
        lambda w, d, h: (w, d, h),
        lambda w, d, h: (d, w, h),
        lambda w, d, h: (h, d, w),
        lambda w, d, h: (d, h, w),
        lambda w, d, h: (w, h, d),
        lambda w, d, h: (h, w, d),
    )

    def __init__(self, inst):
        super().__init__(inst)
        self.name = 'solver_s336733_s343674'
        self.items: Dict[str, ItemData] = {}
        self.vehicles: List[VehicleData] = []
        self.start_time = 0.0
        self.time_limit = 540.0

    def solve(self):
        self.start_time = time.monotonic()
        self._load_data()

        orderings = self._build_orderings()
        vehicle_modes = ('balanced', 'cheap', 'efficient', 'small')
        if len(self.items) > 300:
            orderings = [entry for entry in orderings if entry[0] == 'volume']
            vehicle_modes = ('balanced',)
        elif len(self.items) > 100:
            orderings = [entry for entry in orderings if entry[0] in ('volume', 'base_area')]
            vehicle_modes = ('balanced', 'cheap')

        best_bins = None
        best_cost = math.inf

        if len(self.items) > 300:
            # Large real-world instances need a very fast constructive phase.
            # Keep all placements on the floor; this is conservative for gravity and robust.
            for order_name, ordered_items in orderings:
                if self._time_exceeded():
                    break
                for mode in vehicle_modes:
                    bins = self._pack_items_floor(ordered_items, mode)
                    if bins is None:
                        continue
                    valid, _ = self.validate_solution(bins)
                    if valid:
                        cost = self._total_cost(bins)
                        if cost + EPS < best_cost:
                            best_cost = cost
                            best_bins = bins
            if best_bins is None:
                best_bins = self._single_item_fallback(list(self.items.values()))
            self._bins_to_solution(best_bins)
            valid, reason = self.validate_solution(best_bins)
            if not valid:
                raise RuntimeError(f'Internal validation failed before writing solution: {reason}')
            self.write_solution_to_file()
            return

        for order_name, ordered_items in orderings:
            if self._time_exceeded():
                break
            for mode in vehicle_modes:
                if self._time_exceeded():
                    break
                bins = self._pack_items(ordered_items, mode)
                if bins is None:
                    continue
                valid, _ = self.validate_solution(bins)
                if not valid:
                    continue
                # Local re-insertion is intentionally skipped in the submitted version: the
                # constructive multi-start heuristic is faster and more predictable.
                cost = self._total_cost(bins)
                if cost + EPS < best_cost:
                    best_cost = cost
                    best_bins = bins

        if best_bins is None:
            # Last-resort deterministic construction: one item per feasible vehicle.
            best_bins = self._single_item_fallback(list(self.items.values()))
            valid, reason = self.validate_solution(best_bins)
            if not valid:
                raise RuntimeError(f'Unable to construct a feasible solution: {reason}')

        self._bins_to_solution(best_bins)
        valid, reason = self.validate_solution(best_bins)
        if not valid:
            raise RuntimeError(f'Internal validation failed before writing solution: {reason}')
        self.write_solution_to_file()

    def _load_data(self):
        self.items = {}
        for item_id, row in self.inst.df_items.iterrows():
            width = float(row['width'])
            depth = float(row['depth'])
            height = float(row['height'])
            weight = float(row['weight'])
            value = float(row['value'])
            allowed = ''.join(ch for ch in str(row['allowedRotations']) if ch in '012345')
            if not allowed:
                allowed = '0'
            self.items[str(item_id)] = ItemData(
                id=str(item_id), width=width, depth=depth, height=height,
                weight=weight, value=value, allowed=allowed,
                volume=width * depth * height, base_area=width * depth,
                max_dim=max(width, depth, height),
            )

        self.vehicles = []
        for vehicle_type, row in self.inst.df_vehicles.iterrows():
            width = float(row['width'])
            depth = float(row['depth'])
            height = float(row['height'])
            max_weight = float(row['maxWeight'])
            cost = float(row['cost'])
            raw_max_value = row['maxValue'] if 'maxValue' in row else math.inf
            max_value = math.inf if pd.isna(raw_max_value) else float(raw_max_value)
            gravity = float(row['gravityStrength']) if 'gravityStrength' in row and not pd.isna(row['gravityStrength']) else 0.0
            self.vehicles.append(VehicleData(
                type=str(vehicle_type), width=width, depth=depth, height=height,
                max_weight=max_weight, cost=cost, max_value=max_value,
                gravity=gravity, volume=width * depth * height,
            ))

    def _build_orderings(self):
        items = list(self.items.values())
        key_id = lambda it: it.id
        return [
            ('volume', sorted(items, key=lambda it: (-it.volume, -it.weight, key_id(it)))),
            ('base_area', sorted(items, key=lambda it: (-it.base_area, -it.height, key_id(it)))),
            ('height', sorted(items, key=lambda it: (-it.height, -it.volume, key_id(it)))),
            ('weight', sorted(items, key=lambda it: (-it.weight, -it.volume, key_id(it)))),
            ('value', sorted(items, key=lambda it: (-it.value, -it.volume, key_id(it)))),
            ('few_rot', sorted(items, key=lambda it: (len(set(it.allowed)), -it.volume, key_id(it)))),
            ('mixed', sorted(items, key=lambda it: (-(it.volume * (1.0 + it.max_dim) * (1.0 + it.weight) / max(1, len(set(it.allowed)))), key_id(it)))),
        ]


    def _pack_items_floor(self, ordered_items: Sequence[ItemData], vehicle_mode: str) -> Optional[List[BinState]]:
        bins: List[BinState] = []
        for item in ordered_items:
            best = None
            for bin_state in bins:
                if item.weight > bin_state.vehicle.max_weight - bin_state.used_weight + EPS:
                    continue
                if item.value > bin_state.vehicle.max_value - bin_state.used_value + EPS:
                    continue
                for box, score in self._candidate_boxes_floor(item, bin_state):
                    score -= 100.0 * ((bin_state.used_volume + item.volume) / max(bin_state.vehicle.volume, EPS))
                    if best is None or score < best[1]:
                        best = (bin_state, score, box)
            if best is not None:
                self._add_box(best[0], best[2])
                continue
            new_bin = self._open_best_vehicle_for_item_floor(item, len(bins), vehicle_mode)
            if new_bin is None:
                return None
            bins.append(new_bin)
        return bins

    def _candidate_boxes_floor(self, item: ItemData, bin_state: BinState):
        points = sorted(
            {self._round_point((x, y, 0.0)) for x, y, z in bin_state.candidate_points if abs(z) <= EPS},
            key=lambda p: (p[1], p[0])
        )[:120]
        if not points:
            points = [(0.0, 0.0, 0.0)]
        for orient in sorted(set(int(ch) for ch in item.allowed if ch in '012345')):
            w, d, h = self._rotated_dims(item, orient)
            for x, y, _z in points:
                box = Box(
                    id=item.id,
                    x1=x, y1=y, z1=0.0,
                    x2=x + d, y2=y + w, z2=h,
                    w=w, d=d, h=h, orient=orient,
                    weight=item.weight, value=item.value,
                )
                ok, support_area = self._can_place_box(bin_state, box, item)
                if ok:
                    yield box, self._placement_score(bin_state, box, support_area)

    def _open_best_vehicle_for_item_floor(self, item: ItemData, idx: int, mode: str) -> Optional[BinState]:
        best = None
        for vehicle in self.vehicles:
            if item.weight > vehicle.max_weight + EPS or item.value > vehicle.max_value + EPS:
                continue
            bin_state = BinState(idx=idx, vehicle=vehicle)
            candidates = list(self._candidate_boxes_floor(item, bin_state))
            if not candidates:
                continue
            box, place_score = min(candidates, key=lambda pair: pair[1])
            unused_volume = max(0.0, vehicle.volume - item.volume)
            cost_per_volume = vehicle.cost / max(vehicle.volume, EPS)
            if mode == 'efficient':
                score = cost_per_volume * 1e9 + vehicle.cost + unused_volume * 1e-7 + place_score
            else:
                score = vehicle.cost * 100.0 + cost_per_volume * 1e8 + unused_volume * 1e-6 + place_score
            if best is None or score < best[0]:
                best = (score, bin_state, box)
        if best is None:
            return None
        _, bin_state, box = best
        self._add_box(bin_state, box)
        return bin_state

    def _pack_items(self, ordered_items: Sequence[ItemData], vehicle_mode: str) -> Optional[List[BinState]]:
        bins: List[BinState] = []
        for item in ordered_items:
            placement = self._find_best_existing_placement(item, bins)
            if placement is not None:
                bin_state, box, _score = placement
                self._add_box(bin_state, box)
                continue

            new_bin = self._open_best_vehicle_for_item(item, len(bins), vehicle_mode)
            if new_bin is None:
                return None
            bins.append(new_bin)
        return bins

    def _find_best_existing_placement(self, item: ItemData, bins: Sequence[BinState]):
        best = None
        candidate_bins = sorted(
            bins,
            key=lambda b: (b.vehicle.cost, -(b.vehicle.max_weight - b.used_weight), len(b.boxes))
        )
        if len(self.items) > 300 and len(candidate_bins) > 120:
            candidate_bins = candidate_bins[-60:] + candidate_bins[:60]
        for bin_state in candidate_bins:
            if item.weight > bin_state.vehicle.max_weight - bin_state.used_weight + EPS:
                continue
            if item.value > bin_state.vehicle.max_value - bin_state.used_value + EPS:
                continue
            for box, score in self._candidate_boxes_for_bin(item, bin_state):
                fill = (bin_state.used_volume + item.volume) / max(bin_state.vehicle.volume, EPS)
                score = score - 500.0 * fill
                if best is None or score < best[2]:
                    best = (bin_state, box, score)
        return best

    def _open_best_vehicle_for_item(self, item: ItemData, idx: int, mode: str) -> Optional[BinState]:
        best = None
        for vehicle in self.vehicles:
            if item.weight > vehicle.max_weight + EPS or item.value > vehicle.max_value + EPS:
                continue
            bin_state = BinState(idx=idx, vehicle=vehicle)
            candidates = list(self._candidate_boxes_for_bin(item, bin_state))
            if not candidates:
                continue
            box, place_score = min(candidates, key=lambda pair: pair[1])
            unused_volume = max(0.0, vehicle.volume - item.volume)
            cost_per_volume = vehicle.cost / max(vehicle.volume, EPS)
            weight_slack = max(0.0, vehicle.max_weight - item.weight)
            value_slack = 0.0 if math.isinf(vehicle.max_value) else max(0.0, vehicle.max_value - item.value)
            gravity_penalty = vehicle.gravity * 0.02
            if mode == 'cheap':
                vehicle_score = vehicle.cost * 1000.0 + unused_volume * 1e-6 + place_score
            elif mode == 'efficient':
                vehicle_score = cost_per_volume * 1e9 + vehicle.cost + unused_volume * 1e-7 + place_score
            elif mode == 'small':
                vehicle_score = unused_volume * 1e-5 + vehicle.cost * 10.0 + place_score
            else:
                vehicle_score = (
                    vehicle.cost * 100.0
                    + cost_per_volume * 1e8
                    + unused_volume * 1e-6
                    - min(weight_slack, vehicle.max_weight) * 1e-5
                    - min(value_slack, vehicle.max_value if not math.isinf(vehicle.max_value) else 0.0) * 1e-5
                    + gravity_penalty
                    + place_score
                )
            if best is None or vehicle_score < best[0]:
                best = (vehicle_score, bin_state, box)
        if best is None:
            return None
        _, bin_state, box = best
        self._add_box(bin_state, box)
        return bin_state

    def _candidate_boxes_for_bin(self, item: ItemData, bin_state: BinState):
        points = self._sorted_candidate_points(bin_state)
        dims_seen = set()
        for orient in sorted(set(int(ch) for ch in item.allowed if ch in '012345')):
            w, d, h = self._rotated_dims(item, orient)
            dims_key = (round(w, ROUND_DIGITS), round(d, ROUND_DIGITS), round(h, ROUND_DIGITS))
            if (orient, dims_key) in dims_seen:
                continue
            dims_seen.add((orient, dims_key))
            for x, y, z in points:
                box = Box(
                    id=item.id,
                    x1=x, y1=y, z1=z,
                    x2=x + d, y2=y + w, z2=z + h,
                    w=w, d=d, h=h, orient=orient,
                    weight=item.weight, value=item.value,
                )
                ok, support_area = self._can_place_box(bin_state, box, item)
                if ok:
                    yield box, self._placement_score(bin_state, box, support_area)

    def _sorted_candidate_points(self, bin_state: BinState) -> List[Tuple[float, float, float]]:
        points = set(bin_state.candidate_points)
        # Add grid-like points from existing box faces to improve packing and stacking.
        xs = {0.0}
        ys = {0.0}
        zs = {0.0}
        for box in bin_state.boxes:
            xs.update((box.x1, box.x2))
            ys.update((box.y1, box.y2))
            zs.update((box.z1, box.z2))
        # Keep the generated grid moderate.
        if len(bin_state.boxes) <= 25 and len(xs) * len(ys) * len(zs) <= 1200:
            for z in zs:
                for y in ys:
                    for x in xs:
                        points.add(self._round_point((x, y, z)))
        filtered = []
        v = bin_state.vehicle
        for x, y, z in points:
            if x >= -EPS and y >= -EPS and z >= -EPS and x <= v.depth + EPS and y <= v.width + EPS and z <= v.height + EPS:
                filtered.append(self._round_point((max(0.0, x), max(0.0, y), max(0.0, z))))
        ordered = sorted(set(filtered), key=lambda p: (p[2], p[1], p[0]))
        return ordered[:120]

    def _placement_score(self, bin_state: BinState, box: Box, support_area: float) -> float:
        current_max_x = max((b.x2 for b in bin_state.boxes), default=0.0)
        current_max_y = max((b.y2 for b in bin_state.boxes), default=0.0)
        current_max_z = max((b.z2 for b in bin_state.boxes), default=0.0)
        growth_x = max(0.0, box.x2 - current_max_x)
        growth_y = max(0.0, box.y2 - current_max_y)
        growth_z = max(0.0, box.z2 - current_max_z)
        support_ratio = 1.0 if box.z1 <= EPS else support_area / max(box.base_area, EPS)
        return (
            box.z1 * 1000.0
            + box.y1 * 5.0
            + box.x1 * 3.0
            + box.z2 * 20.0
            + growth_z * 50.0
            + growth_y * 5.0
            + growth_x * 3.0
            - support_ratio * 100.0
        )

    def _can_place_box(self, bin_state: BinState, box: Box, item: Optional[ItemData] = None) -> Tuple[bool, float]:
        v = bin_state.vehicle
        if box.x1 < -EPS or box.y1 < -EPS or box.z1 < -EPS:
            return False, 0.0
        if box.x2 > v.depth + EPS or box.y2 > v.width + EPS or box.z2 > v.height + EPS:
            return False, 0.0
        if box.weight + bin_state.used_weight > v.max_weight + EPS:
            return False, 0.0
        if box.value + bin_state.used_value > v.max_value + EPS:
            return False, 0.0
        if item is not None and str(box.orient) not in item.allowed:
            return False, 0.0
        for other in bin_state.boxes:
            if self._boxes_overlap(box, other):
                return False, 0.0
        support_area = self._support_area(bin_state.boxes, box)
        if box.z1 <= EPS:
            return True, box.base_area
        required = box.base_area * (v.gravity / 100.0)
        return support_area + EPS >= required, support_area

    def _support_area(self, boxes: Sequence[Box], box: Box) -> float:
        if box.z1 <= EPS:
            return box.base_area
        support_area = 0.0
        for other in boxes:
            if abs(other.z2 - box.z1) < 1e-6:
                dx = max(0.0, min(box.x2, other.x2) - max(box.x1, other.x1))
                dy = max(0.0, min(box.y2, other.y2) - max(box.y1, other.y1))
                support_area += dx * dy
        return support_area

    def _add_box(self, bin_state: BinState, box: Box):
        box = self._round_box(box)
        bin_state.boxes.append(box)
        bin_state.used_weight += box.weight
        bin_state.used_value += box.value
        bin_state.used_volume += box.w * box.d * box.h
        new_points = (
            (box.x2, box.y1, box.z1), (box.x1, box.y2, box.z1), (box.x1, box.y1, box.z2),
            (box.x2, box.y2, box.z1), (box.x2, box.y1, box.z2), (box.x1, box.y2, box.z2),
        )
        for point in new_points:
            rounded = self._round_point(point)
            if rounded[0] <= bin_state.vehicle.depth + EPS and rounded[1] <= bin_state.vehicle.width + EPS and rounded[2] <= bin_state.vehicle.height + EPS:
                bin_state.candidate_points.add(rounded)

    def _improve_bins(self, bins: List[BinState], mode: str) -> List[BinState]:
        best = self._clone_bins(bins)
        changed = True
        rounds = 0
        while changed and rounds < 2 and not self._time_exceeded():
            changed = False
            rounds += 1
            order = sorted(range(len(best)), key=lambda i: (best[i].used_volume / max(best[i].vehicle.volume, EPS), -best[i].vehicle.cost))
            for remove_idx in order:
                if len(best) <= 1 or remove_idx >= len(best):
                    continue
                trial = self._clone_bins(best)
                removed = trial.pop(remove_idx)
                for i, b in enumerate(trial):
                    b.idx = i
                items_to_reinsert = [self.items[b.id] for b in sorted(removed.boxes, key=lambda bx: -bx.w * bx.d * bx.h)]
                success = True
                for item in items_to_reinsert:
                    placement = self._find_best_existing_placement(item, trial)
                    if placement is None:
                        success = False
                        break
                    bin_state, box, _ = placement
                    self._add_box(bin_state, box)
                if success:
                    valid, _ = self.validate_solution(trial)
                    if valid and self._total_cost(trial) + EPS <= self._total_cost(best):
                        best = trial
                        changed = True
                        break
        return best

    def _single_item_fallback(self, items: Sequence[ItemData]) -> List[BinState]:
        bins = []
        for item in items:
            new_bin = self._open_best_vehicle_for_item(item, len(bins), 'cheap')
            if new_bin is None:
                raise RuntimeError(f'Item {item.id} cannot fit in any vehicle')
            bins.append(new_bin)
        return bins

    def validate_solution(self, bins: Sequence[BinState]) -> Tuple[bool, str]:
        seen = []
        for expected_idx, bin_state in enumerate(bins):
            if bin_state.idx != expected_idx:
                return False, f'idx_vehicle not consecutive at {expected_idx}'
            total_weight = 0.0
            total_value = 0.0
            for i, box in enumerate(bin_state.boxes):
                if box.id not in self.items:
                    return False, f'unknown item {box.id}'
                item = self.items[box.id]
                if str(box.orient) not in item.allowed:
                    return False, f'rotation not allowed for {box.id}'
                expected_dims = self._rotated_dims(item, box.orient)
                if any(abs(a - b) > 1e-5 for a, b in zip((box.w, box.d, box.h), expected_dims)):
                    return False, f'wrong rotated dimensions for {box.id}'
                if box.x1 < -EPS or box.y1 < -EPS or box.z1 < -EPS:
                    return False, f'negative coordinate for {box.id}'
                v = bin_state.vehicle
                if box.x2 > v.depth + EPS or box.y2 > v.width + EPS or box.z2 > v.height + EPS:
                    return False, f'out of bounds {box.id}'
                for other in bin_state.boxes[:i]:
                    if self._boxes_overlap(box, other):
                        return False, f'overlap {box.id} with {other.id}'
                support_area = self._support_area(bin_state.boxes, box)
                required = 0.0 if box.z1 <= EPS else box.base_area * (v.gravity / 100.0)
                if support_area + EPS < required:
                    return False, f'gravity {box.id}'
                total_weight += box.weight
                total_value += box.value
                seen.append(box.id)
            if total_weight > bin_state.vehicle.max_weight + EPS:
                return False, f'weight exceeded vehicle {bin_state.idx}'
            if total_value > bin_state.vehicle.max_value + EPS:
                return False, f'value exceeded vehicle {bin_state.idx}'
        if len(seen) != len(set(seen)):
            return False, 'duplicate item placement'
        expected = set(self.items.keys())
        actual = set(seen)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            return False, f'items mismatch missing={missing[:5]} extra={extra[:5]}'
        return True, 'ok'

    def _bins_to_solution(self, bins: Sequence[BinState]):
        self.sol = {key: [] for key in self.sol}
        self.idx_vehicle = len(bins)
        for new_idx, bin_state in enumerate(bins):
            for box in bin_state.boxes:
                self.sol['type_vehicle'].append(bin_state.vehicle.type)
                self.sol['idx_vehicle'].append(new_idx)
                self.sol['id_item'].append(box.id)
                self.sol['x_origin'].append(round(box.x1, ROUND_DIGITS))
                self.sol['y_origin'].append(round(box.y1, ROUND_DIGITS))
                self.sol['z_origin'].append(round(box.z1, ROUND_DIGITS))
                self.sol['orient'].append(int(box.orient))

    def _rotated_dims(self, item: ItemData, orient: int) -> Tuple[float, float, float]:
        return self.ROTATIONS[int(orient)](item.width, item.depth, item.height)

    def _boxes_overlap(self, a: Box, b: Box) -> bool:
        return (
            max(a.x1, b.x1) < min(a.x2, b.x2) - EPS and
            max(a.y1, b.y1) < min(a.y2, b.y2) - EPS and
            max(a.z1, b.z1) < min(a.z2, b.z2) - EPS
        )

    def _round_point(self, point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return tuple(round(max(0.0, float(coord)), ROUND_DIGITS) for coord in point)

    def _round_box(self, box: Box) -> Box:
        return Box(
            id=box.id,
            x1=round(box.x1, ROUND_DIGITS), y1=round(box.y1, ROUND_DIGITS), z1=round(box.z1, ROUND_DIGITS),
            x2=round(box.x2, ROUND_DIGITS), y2=round(box.y2, ROUND_DIGITS), z2=round(box.z2, ROUND_DIGITS),
            w=box.w, d=box.d, h=box.h, orient=box.orient, weight=box.weight, value=box.value,
        )

    def _clone_bins(self, bins: Sequence[BinState]) -> List[BinState]:
        cloned = [b.clone() for b in bins]
        for i, b in enumerate(cloned):
            b.idx = i
        return cloned

    def _total_cost(self, bins: Sequence[BinState]) -> float:
        return sum(bin_state.vehicle.cost for bin_state in bins)

    def _time_exceeded(self) -> bool:
        return time.monotonic() - self.start_time > self.time_limit

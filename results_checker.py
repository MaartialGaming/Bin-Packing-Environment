import os
import pandas as pd
from instances import Instance

if __name__ == '__main__':
    dataset_name = 'DatasetA'
    solver_name = 'solver_000000'
    inst = Instance(dataset_name)

    # -------------------------
    # Load data
    # -------------------------
    items = inst.df_items
    vehicles = inst.df_vehicles
    solution = pd.read_csv(os.path.join('results', f'sol_{dataset_name}_{solver_name}.csv'))

    # -------------------------
    # Helpers
    # -------------------------
    def get_dims(item, orient):
        w, d, h = item["width"], item["depth"], item["height"]

        rotations = [
            (w, d, h),
            (d, w, h),
            (h, d, w),
            (d, h, w),
            (w, h, d),
            (h, w, d),
        ]
        return rotations[orient]


    def overlap_1d(a_min, a_max, b_min, b_max):
        return max(a_min, b_min) < min(a_max, b_max)


    def boxes_overlap(a, b):
        return (
            overlap_1d(a["x1"], a["x2"], b["x1"], b["x2"]) and
            overlap_1d(a["y1"], a["y2"], b["y1"], b["y2"]) and
            overlap_1d(a["z1"], a["z2"], b["z1"], b["z2"])
        )


    # -------------------------
    # Prepare lookups
    # -------------------------
    required_columns = {
        "type_vehicle",
        "idx_vehicle",
        "id_item",
        "x_origin",
        "y_origin",
        "z_origin",
        "orient",
    }

    missing_columns = required_columns - set(solution.columns)
    if missing_columns:
        print(f"MISSING COLUMNS: {', '.join(sorted(missing_columns))}")
        solution = solution.reindex(columns=list(required_columns))

    numeric_columns = ["idx_vehicle", "x_origin", "y_origin", "z_origin", "orient"]
    for col in numeric_columns:
        solution[col] = pd.to_numeric(solution[col], errors="coerce")

    items_dict = items.to_dict("index")
    vehicles_dict = vehicles.to_dict("index")

    groups = solution.groupby("idx_vehicle")

    total_cost = 0
    feasible = not missing_columns
    placed_item_ids = set()

    print("\n================ SOLUTION CHECK ================\n")

    bad_numeric = solution[numeric_columns].isna().any(axis=1)
    if bad_numeric.any():
        print(f"INVALID NUMERIC VALUES in rows: {bad_numeric[bad_numeric].index.tolist()}")
        feasible = False

    vehicle_indices = sorted(solution["idx_vehicle"].dropna().unique())
    expected_indices = list(range(len(vehicle_indices)))
    if vehicle_indices and vehicle_indices != expected_indices:
        print(f"NON-CONSECUTIVE VEHICLE INDICES: {vehicle_indices}")
        feasible = False

    for vidx, group in groups:
        vehicle_type = group.iloc[0]['type_vehicle']
        if group["type_vehicle"].nunique(dropna=False) != 1:
            print(f"\nVehicle {vidx}:")
            print("MIXED VEHICLE TYPES")
            feasible = False
            continue

        if vehicle_type not in vehicles_dict:
            print(f"\nVehicle {vidx}:")
            print(f"UNKNOWN VEHICLE TYPE: {vehicle_type}")
            feasible = False
            continue

        vehicle = vehicles_dict[vehicle_type]

        vehicle_ok = True
        placed_boxes = []
        total_weight = 0
        total_value = 0

        total_cost += vehicle["cost"]

        for _, row in group.iterrows():
            item_id = row["id_item"]
            if item_id in placed_item_ids:
                print(f"DUPLICATE ITEM: {item_id}")
                vehicle_ok = False
            placed_item_ids.add(item_id)

            if item_id not in items_dict:
                print(f"UNKNOWN ITEM: {item_id}")
                vehicle_ok = False
                continue

            item = items_dict[item_id]

            orient = int(row["orient"])
            if orient < 0 or orient > 5:
                print(f"INVALID ORIENTATION: {item_id} orient={orient}")
                vehicle_ok = False
                continue

            if str(orient) not in str(item["allowedRotations"]):
                print(f"FORBIDDEN ROTATION: {item_id} orient={orient}, allowed={item['allowedRotations']}")
                vehicle_ok = False

            w, d, h = get_dims(item, orient)

            x, y, z = row["x_origin"], row["y_origin"], row["z_origin"]

            box = {
                "id": item_id,
                "x1": x, "y1": y, "z1": z,
                "x2": x + d, "y2": y + w, "z2": z + h,
                "base_area": w * d
            }

            # -------------------------
            # Bounds
            # -------------------------
            if box["x1"] < 0 or \
            box["y1"] < 0 or \
            box["z1"] < 0 or \
            box["x2"] > vehicle["depth"] or \
            box["y2"] > vehicle["width"] or \
            box["z2"] > vehicle["height"]:
                print(f"\nVehicle {vidx} ({vehicle_type}):")
                print(f"OUT OF BOUNDS: {item_id}")
                vehicle_ok = False

            # -------------------------
            # Overlap
            # -------------------------
            for other in placed_boxes:
                if boxes_overlap(box, other):
                    print(f"\nVehicle {vidx} ({vehicle_type})")
                    print(f"OVERLAP: {item_id} with {other['id']}")
                    vehicle_ok = False

            placed_boxes.append(box)

            total_weight += item["weight"]
            total_value += item["value"]

        # -------------------------
        # Weight
        # -------------------------
        if total_weight > vehicle["maxWeight"]:
            print(f"\nVehicle {vidx} ({vehicle_type})")
            print(f" WEIGHT: {total_weight:.2f} > {vehicle['maxWeight']}")
            vehicle_ok = False

        # -------------------------
        # Value
        # -------------------------
        if total_value > vehicle["maxValue"]:
            print(f"\nVehicle {vidx} ({vehicle_type})")
            print(f"VALUE: {total_value:.2f} > {vehicle['maxValue']}")
            vehicle_ok = False

        # -------------------------
        # Gravity
        # -------------------------
        gravity = vehicle["gravityStrength"]

        for i, box in enumerate(placed_boxes):
            if box["z1"] == 0:
                continue

            support_area = 0

            for j, other in enumerate(placed_boxes):
                if i == j:
                    continue

                if abs(other["z2"] - box["z1"]) < 1e-6:
                    dx = max(0, min(box["x2"], other["x2"]) - max(box["x1"], other["x1"]))
                    dy = max(0, min(box["y2"], other["y2"]) - max(box["y1"], other["y1"]))
                    support_area += dx * dy

            required = box["base_area"] * (gravity / 100.0)

            if support_area < required:
                print(f"\nVehicle {vidx} ({vehicle_type})")
                print(f"GRAVITY: {box['id']} supported {support_area:.1f} < {required:.1f}")
                vehicle_ok = False

        if not vehicle_ok:
            feasible = False
    
    # -------------------------
    # Check that all items are placed
    # -------------------------
    all_items = set(items.index)
    missing_items = all_items - placed_item_ids

    if missing_items:
        print(f"MISSING ITEMS: {', '.join(sorted(missing_items))}")
        feasible = False

    # -------------------------
    # Final summary
    # -------------------------
    print("=============== SUMMARY ===============")
    print(f"Total cost: {total_cost:.2f}")

    if feasible:
        print("FEASIBLE solution")
    else:
        print("INFEASIBLE solution")

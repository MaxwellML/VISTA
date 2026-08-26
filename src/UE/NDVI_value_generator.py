from pathlib import Path
import csv
import math
import random
import os


SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_CSV = SCRIPT_DIR / "foliage_instances.csv"
OUTPUT_CSV = SCRIPT_DIR / "foliage_instances.csv"

# Fixed seed means running the script again produces
# exactly the same NDVI values.
RANDOM_SEED = 42
rng = random.Random(RANDOM_SEED)


# ------------------------------------------------------------
# NDVI distributions
# ------------------------------------------------------------

NDVI_CLASSES = {
    "LARGE_TREE": {
        "mean": 0.63,
        "variance": 0.0324,   
    },

    "MEDIUM_TREE": {
        "mean": 0.63,
        "variance": 0.0324,
    },

    "SMALL_TREE": {
        "mean": 0.63,
        "variance": 0.0324,
    },

    "GRASS_TREE": {
        "mean": 0.55,
        "variance": 0.0256,   
    },

    "ROCK": {
        "mean": 0.03,
        "variance": 0.0025,
    },

    "LOG": {
        "mean": 0.10,
        "variance": 0.0025,
    },

    "SCATTER": None,
}


def classify_mesh(mesh_name):
    if mesh_name.startswith("SM_Tree_L_"):
        return "LARGE_TREE"

    if mesh_name.startswith("SM_Tree_M_"):
        return "MEDIUM_TREE"

    if mesh_name.startswith("SM_Tree_S_"):
        return "SMALL_TREE"

    if mesh_name.startswith("SM_GrassTree_"):
        return "GRASS_TREE"

    if mesh_name.startswith("SM_Rock_"):
        return "ROCK"

    if mesh_name.startswith("SM_Log_"):
        return "LOG"

    if mesh_name.startswith("SM_Scatter_"):
        return "SCATTER"

    return "UNKNOWN"

#normally distribute NDVIs 
def generate_ndvi(class_name):
    params = NDVI_CLASSES.get(class_name)

    if params is None:
        return None

    mean = params["mean"]
    variance = params["variance"]

    standard_deviation = math.sqrt(variance)

    ndvi = rng.gauss(mean, standard_deviation)

    return max(-1.0, min(1.0, ndvi))


def main():
    # --------------------------------------------------------
    # Read the existing CSV completely before modifying it
    # --------------------------------------------------------

    with INPUT_CSV.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        rows = list(reader)
        fieldnames = list(reader.fieldnames)


    # --------------------------------------------------------
    # Add/update columns
    # --------------------------------------------------------

    if "ndvi_class" not in fieldnames:
        fieldnames.append("ndvi_class")

    if "NDVI" not in fieldnames:
        fieldnames.append("NDVI")


    class_counts = {}

    for row in rows:
        mesh_name = row["mesh_name"]

        class_name = classify_mesh(mesh_name)
        ndvi = generate_ndvi(class_name)

        row["ndvi_class"] = class_name

        if ndvi is None:
            row["NDVI"] = ""
        else:
            row["NDVI"] = f"{ndvi:.6f}"

        class_counts[class_name] = class_counts.get(class_name, 0) + 1


    # --------------------------------------------------------
    # Write to temporary file first
    #
    # Since INPUT_CSV and OUTPUT_CSV are deliberately the same
    # file, this prevents the original CSV being destroyed if
    # something goes wrong while writing.
    # --------------------------------------------------------

    temp_csv = OUTPUT_CSV.with_suffix(".tmp.csv")

    with temp_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


    #Save to temporary CSV, not original. 
    os.replace(temp_csv, OUTPUT_CSV)

    print()
    print("======================================")
    print("NDVI ASSIGNMENT COMPLETE")
    print("======================================")
    print(f"Instances processed: {len(rows)}")
    print()

    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name:<15} {count:>6}")

    print()
    print(f"Saved to:")
    print(OUTPUT_CSV)
    print("======================================")


if __name__ == "__main__":
    main()
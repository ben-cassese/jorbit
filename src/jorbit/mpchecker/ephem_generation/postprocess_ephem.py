import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: I001
import sqlite3

import numpy as np
from tqdm import tqdm
from glob import glob

from jorbit.mpchecker.contribute_to_ephem import mpc_code_to_number


##########
# merge all of the individually-generated databases into one
##########


def combine_databases(input_pattern, output_db):
    with sqlite3.connect(output_db) as output_conn:
        output_cursor = output_conn.cursor()

        output_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                target_name TEXT PRIMARY KEY,
                chebyshev_coefficients BLOB,
                x0 BLOB,
                v0 BLOB
            )
            """
        )

        for db_file in tqdm(glob(input_pattern)):
            # print(f"Processing {db_file}")
            try:
                with sqlite3.connect(db_file) as input_conn:
                    input_cursor = input_conn.cursor()

                    input_cursor.execute(
                        "SELECT target_name, chebyshev_coefficients, x0, v0 FROM results"
                    )
                    rows = input_cursor.fetchall()

                    # OR IGNORE skips duplicates
                    output_cursor.executemany(
                        "INSERT OR IGNORE INTO results (target_name, chebyshev_coefficients, x0, v0) VALUES (?, ?, ?, ?)",
                        rows,
                    )
                output_conn.commit()
            except Exception:
                pass


combine_databases("perturbers*", "merged_results.db")
combine_databases("db_results/*", "merged_results.db")


##########
# get the list of successfully processed targets
##########

with sqlite3.connect("merged_results.db") as conn:
    cursor = conn.cursor()

    cursor.execute("SELECT target_name FROM results")

    processed_targets = [row[0] for row in cursor.fetchall()]

processed_targets = list(set(processed_targets))


##########
# get the list of all targets we were trying to include
##########

with open("MPCORB.DAT") as f:
    lines = f.readlines()[43:]

# all_targets = [line.split()[0] for line in lines ]# this broke, invalid split somewhere
all_targets = []
for i in tqdm(range(len(lines))):
    try:
        target = lines[i].split()[0]
        all_targets.append(target)
    except Exception:
        print(f"Error on line {i}")

all_targets = [mpc_code_to_number(target) for target in all_targets]
all_targets = list(set(all_targets))


##########
# get the list of targets that Horizons didn't know about
##########

horizon_missing_files = glob("db_results/*.txt")
not_in_horizons = []
for f in tqdm(horizon_missing_files):
    with open(f) as f:
        not_in_horizons.extend(f.read().splitlines())
not_in_horizons = list(set(not_in_horizons))


##########
# get the list of targets that we could have but failed to process for some reason
##########
missing_targets = list(set(all_targets) - set(processed_targets) - set(not_in_horizons))

missing_lines = []
# 43 was the first line of data in the table
for i in tqdm(range(43, len(lines))):
    try:
        target = lines[i].split()[0]
        t = mpc_code_to_number(target)
        if t in missing_targets:
            missing_lines.append(i)

    except Exception:
        print(f"Error on line {i}")
missing_lines = list(set(missing_lines))


##########
# make a new file with the targets that we missed
##########

with open("missed_targets.DAT", "w") as f:
    for i in missing_lines:
        f.write(lines[i])


batch_size = 100
n_jobs = 500
end = False
with open("redo_jobs.txt", "w") as f:
    for i in range(n_jobs):
        start = i * batch_size
        stop = (i + 1) * batch_size
        if stop > len(missing_lines):
            stop = len(missing_lines)
            end = True
        f.write(f"uv run python contribute_to_ephem.py {start} {stop}\n")
        if end:
            print("last line")
            break
    if not end:
        raise ValueError(
            "not enough jobs to cover all missing lines w/ this batch size"
        )


##########
# post-process the final database into chunk files
# (this will take a lot of memory)
##########
with sqlite3.connect("merged_results.db") as conn:
    cursor = conn.cursor()

    cursor.execute("SELECT target_name, chebyshev_coefficients, x0, v0 FROM results")
    results = cursor.fetchall()

sorted_results = sorted(results, key=lambda x: x[0])
coeffs = np.array([np.frombuffer(x[1]).reshape(11, 2, 244) for x in sorted_results])
x0 = np.array([np.frombuffer(x[2]) for x in sorted_results])
v0 = np.array([np.frombuffer(x[3]) for x in sorted_results])
names = [x[0] for x in sorted_results]

jnp.save("data_products/x0.npy", x0)
jnp.save("data_products/v0.npy", v0)
np.save("data_products/names.npy", np.array(names))
for i in range(coeffs.shape[-1]):
    jnp.save(
        f"data_products/chebyshev_coeffs_fwd_{str(i).zfill(3)}.npy", coeffs[:, :, :, i]
    )

from pathlib import Path

source = Path("src/pipeline_real_tiles_fixed.py")
s = source.read_text()

parser = "a = ap.parse_args()"
rng_line = "rng = np.random.default_rng(a.seed)"
arrival_line = "due += rng.exponential(1.0 / rate)"

for needle in (parser, rng_line, arrival_line):
    assert s.count(needle) == 1, (
        f"Expected exactly one occurrence: {needle}"
    )

s = s.replace(
    parser,
    '''ap.add_argument(
    "--arrival-mode",
    choices=["periodic", "poisson-matched", "poisson"],
    default="poisson-matched"
)
''' + parser,
    1
)

schedule = '''

# Precompute arrival gaps before request submission.
# This allows periodic and random traces to have the
# same offered rate over the same number of requests.
interarrivals = np.zeros(a.requests, dtype=np.float64)

for j in range(1, a.requests):
    phase = j // a.phase_requests
    rate_j = phase_rates[phase]

    if a.arrival_mode == "periodic":
        interarrivals[j] = 1.0 / rate_j
    else:
        interarrivals[j] = rng.exponential(1.0 / rate_j)

if a.arrival_mode == "poisson-matched":
    for phase, rate_j in enumerate(phase_rates):
        indices = np.array([
            j for j in range(1, a.requests)
            if j // a.phase_requests == phase
        ], dtype=int)

        if len(indices) == 0:
            continue

        sampled_total = interarrivals[indices].sum()
        desired_total = len(indices) / rate_j

        interarrivals[indices] *= (
            desired_total / sampled_total
        )

print(
    "Arrival mode:", a.arrival_mode,
    "Planned duration s:",
    round(interarrivals.sum(), 3),
    flush=True
)
'''

s = s.replace(rng_line, rng_line + schedule, 1)
s = s.replace(
    arrival_line,
    "due += interarrivals[i]",
    1
)

dest = Path("src/pipeline_arrival_matched.py")
dest.write_text(s)

print("Created:", dest)
print("Original source unchanged:", source)

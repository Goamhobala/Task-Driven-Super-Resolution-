#!/usr/bin/env python3
"""Submit an InstaRoad experiment as a Lightning *background Job* via the
Lightning SDK. Unlike job.sh (which detaches inside the current Studio), a
Lightning Job runs on its own machine that starts up, runs the command, and
tears down on its own — handy for long single-GPU refits you don't want tied to
your interactive session.

Requires the Lightning SDK (installed in Lightning Studios by default):
    pip install -U lightning-sdk        # if missing

Examples
--------
# Full tune->fit->bench for one SR arm, on an L4:
python scripts/LightningStudio/submit_job.py --machine L4 \
    -- run_both sr/r2a_all.sh SEED=0

# Just the fit stage on the default machine:
python scripts/LightningStudio/submit_job.py -- run sr/r2a_all.sh STAGE=fit SEED=0

# Anything else, as a raw command:
python scripts/LightningStudio/submit_job.py --machine A10G \
    --command "bash scripts/LightningStudio/run_both.sh unet/cdngi.sh"

Notes
-----
* --machine takes a Machine enum name (e.g. T4, L4, A10G, A100). Which types
  your plan can launch, and their names, may differ — see
  https://lightning.ai/docs/overview/scale-with-batch-jobs/sdk
* --teamspace / --user / --studio default to the current Studio's context when
  omitted (the SDK resolves them from the environment inside a Studio).
"""
from __future__ import annotations
import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machine", default="L4",
                   help="Machine enum name (T4, L4, A10G, A100, CPU, ...). Default: L4")
    p.add_argument("--name", default=None, help="Job name (default: derived from the command)")
    p.add_argument("--command", default=None,
                   help="Raw command to run. If omitted, everything after `--` is treated as "
                        "run.sh/run_both.sh/run_pair.sh args.")
    p.add_argument("--teamspace", default=None, help="Teamspace (default: current Studio context)")
    p.add_argument("--user", default=None, help="User/org (default: current Studio context)")
    p.add_argument("--studio", default=None, help="Studio name (default: current Studio)")
    p.add_argument("rest", nargs=argparse.REMAINDER,
                   help="After `--`: <run|run_both|run_pair> <args...>")
    args = p.parse_args()

    # Build the command.
    if args.command:
        command = args.command
    else:
        rest = [a for a in args.rest if a != "--"]
        if not rest:
            p.error("provide --command, or a dispatcher + args after `--` "
                    "(e.g. `-- run_both sr/r2a_all.sh SEED=0`)")
        disp, *disp_args = rest
        if disp not in ("run", "run_both", "run_pair"):
            p.error(f"first token after `--` must be run|run_both|run_pair (got '{disp}')")
        command = "bash scripts/LightningStudio/%s.sh %s" % (disp, " ".join(disp_args))

    if args.name:
        name = args.name
    else:
        # Derive a readable name from the first *.sh token in the command.
        tag = next((t for t in command.split() if t.endswith(".sh") and "run" not in t.split("/")[-1]),
                   "job")
        name = "instaroad-" + tag.split("/")[-1].replace(".sh", "")

    try:
        from lightning_sdk import Studio, Machine, Job
    except ImportError:
        print("ERROR: lightning-sdk not installed. Run:  pip install -U lightning-sdk", file=sys.stderr)
        return 1

    try:
        machine = getattr(Machine, args.machine)
    except AttributeError:
        print(f"ERROR: unknown machine '{args.machine}'. Try T4 / L4 / A10G / A100 / CPU.", file=sys.stderr)
        return 2

    # Resolve the Studio. Inside a running Studio, omitting name/teamspace/user
    # lets the SDK pick up the current context.
    studio_kw = {k: v for k, v in
                 dict(name=args.studio, teamspace=args.teamspace, user=args.user).items()
                 if v is not None}
    studio = Studio(**studio_kw)

    print(f"submitting Lightning Job '{name}' on {args.machine}")
    print(f"  command: {command}")
    job = Job.run(command=command, name=name, machine=machine, studio=studio)
    print(f"  status : {job.status}")
    print("Follow it in the Studio's Jobs panel, or query job.status from the SDK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Contributing

## Keep the data out

Competition terms can disqualify the team, so these are not optional.

1. Keep the data outside this repo, on an encrypted disk image, and point to it:
   ```bash
   export MTAH_DATA_DIR=/path/to/mtah-data
   ```
   Do not unpack it into `vendor/challenge/data/`, even though the official guide says `./data`.
2. Eject the disk whenever AI tools are running.
3. Start AI tools from inside this repo, never from your home folder.
4. Run the real pipeline yourself, in your own terminal, with no AI session running.
5. Logs, PRs and chat show aggregates only: counts and scores, never rows, coordinates or screenshots of data.

`.gitignore` only stops git. It does not stop an AI tool from reading a file.

## Setup

```bash
git clone --recurse-submodules https://github.com/falafelona/more-than-a-hack-2027
make setup
make test
```

If you forgot the flag: `git submodule update --init`.

## Making a change

- Branch from `main` as `<type>/<ws>-<short-desc>`, e.g. `feat/ball-gap-fill`.
  - Types: `feat` `exp` `fix` `docs` `chore`
  - Workstreams: `int` `ball` `dead` `poss` `eval`
- Open a PR. It needs one review, then squash-merge.
- After the freeze on 14 Nov, only `fix` and `docs` PRs.

## Code rules

- Python 3.12, with dependencies pinned exactly in `requirements*.txt`.
- Every threshold lives in `configs/`, chosen on other matches.
- The pipeline is deterministic: fixed seeds, no order-dependent parallelism.
- Never edit `vendor/challenge/`. To update it, bump the pinned commit in a `chore` PR.
- A PR that changes the output adds a row to [experiments/RESULTS.md](experiments/RESULTS.md).

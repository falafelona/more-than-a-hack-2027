## What and why
<!-- One or two sentences. -->

## How to check it
<!-- What should the reviewer run or look at? -->

## Checklist
- [ ] **No match data**, extracts or screenshots of data in this PR
- [ ] No match data was shared with a hosted AI tool
- [ ] Nothing tuned or chosen on the public 5 minutes or this match
- [ ] New thresholds live in `configs/`
- [ ] `vendor/challenge/` untouched
- [ ] Tests pass (`make test`, once it exists)
- [ ] If the output changed: row added to `experiments/RESULTS.md`, and the file passes `validate_submission.py`

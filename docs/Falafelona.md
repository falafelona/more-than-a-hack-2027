# Falafelona: technical document

Deliverable 3. A few pages. Exported to Falafelona.pdf only at submission time (PDFs are gitignored).

Each section has one owner, who writes it and opens a `docs/<ws>-...` PR.
Laila (eval) collects the sections, edits for one voice and exports the PDF (task 5.7b, 16 Nov).
Aggregates only: no rows, coordinates or screenshots of match data.

## 1. Approach
Owner: Dudu + Abdallah (task 4.8)
The pipeline end to end, and why the H/A/D decoder is the centre of it (D13).

### 1.1 Ball track
Owner: Abdallah (task 2.7)
How we clean the ball: false detections removed, gaps filled, and methods we tried and dropped.

### 1.2 Dead ball and restarts
Owner: Chikhani (task 3.8)
How the dead-ball score is built and how restarts are found.

## 2. Assets used (and dropped)
Owner: Tala
Which provided files we use (tracking, Smart Tagging, homography, video) and which we don't, and why.

## 3. Training data (required)
Owner: Laila
The IDSSE dev set: what it is, how it is used for tuning, organiser approval (official issue #16),
and the citation (Bassek et al., Scientific Data 12, 195, 2025). No hand labels on this match.

## 4. Possession rules
Owner: Dudu + Abdallah (task 4.8)
How our labels follow the official possession definitions in CHALLENGE.md.

## 5. Results
Owner: Laila
Dev-set spellF1 on IDSSE, label-free stats on this match, and the baseline (0.2433) for comparison.

## 6. Limitations
Owner: Laila (collects one or two lines from each owner)
What we know is weak, e.g. random ball gaps on the dev set only approximate our real gaps.

## 7. Reproduction
Owner: Tala
One command, the SHA-256 of the file, runtime and machine.
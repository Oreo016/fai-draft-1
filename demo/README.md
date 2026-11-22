# Information sharing demo

A local web demo for showing whether a known report helps a teammate enough to justify its communication cost.

## Run it

From the repository root, install the project dependencies if needed, then start the demo:

```powershell
python -m pip install -e .
python demo/app.py
```

Open <http://127.0.0.1:8765> in a browser. Stop the server with `Ctrl+C`.

## What is included

- **Compare one report:** finds the Scout's first discovered victim report and compares estimated task return when it is sent to the selected teammate versus withheld. It shows the return difference, the message cost, and the remaining value after cost.
- **Play an episode:** runs the actual SAR-Grid and task policies, then lets a viewer play or scrub through the saved grid frames. Map, communication policy, cost, and scenario number can be changed.
- **Quick rescue walkthrough:** switches to a 6 × 6 map with one victim and no hazards, using Full communication by default. This shortens the search and carry-to-evacuation loop so the mechanics are easier to observe. It is a teaching preset, not a research condition.

Episodes use the project's 100-step horizon. A single comparison illustrates the question but cannot establish a general result; change one setting at a time and compare several scenario numbers.

The server binds to `127.0.0.1` only and writes no run artifacts. It needs the repository's project dependencies (`PyYAML` and `Pillow`) and reads the existing default configuration.

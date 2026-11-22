# Receiver-Aware Communication for Search and Rescue

**Foundations of AI course project** · Indian Institute of Technology Gandhinagar  
**Instructor:** Prof. Manisha Padala

We study whether a search-and-rescue team should share an existing report, which teammate should receive it, and whether the expected benefit is worth the communication cost. The project uses a grid-based simulation with three fixed roles: the Scout finds victims, the Medic stabilizes victims and disables hazards, and the Carrier transports stabilized victims to the evacuation cell.

## Project submitters

| Name | Department | Institute | Student ID |
|---|---|---|---|
| Chitiveli Hemcharan Varma | Computer Science | Indian Institute of Technology Gandhinagar | 24110095 |
| Guransh Kaur Saran | Computer Science | Indian Institute of Technology Gandhinagar | 24110127 |
| Kowel P Laloo | Computer Science | Indian Institute of Technology Gandhinagar | 24110168 |
| Omm Arindam | Artificial Intelligence | Indian Institute of Technology Gandhinagar | 24110229 |

## Start here

1. **Read the final report:** [report_final.pdf](report_final.pdf).
2. **Try the interactive demo:** from the repository folder, run:

   ```powershell
   python -m pip install -e .
   python demo/app.py
   ```

   Then open [http://127.0.0.1:8765](http://127.0.0.1:8765). The **Compare one report** page compares the estimated team score when the Scout's discovered victim report is sent to the Medic or Carrier versus withheld. Change the message cost to see how it affects the value of sending. **Play an episode** shows the team moving through the grid; the small rescue walkthrough is a quick way to see a victim evacuated. More setup details are in [demo/README.md](demo/README.md).

The demo's individual comparisons and episodes are illustrations of the method. The research findings and their uncertainty are reported in the PDF; one interactive run does not confirm a hypothesis.

## Reproducing the work

The final report PDF and saved experiment outputs are included in this repository. The full experiment can take about 32 minutes to rerun on the recorded setup, so running it is optional for reviewing the project. The submitted report is available as a PDF; its editable LaTeX source is not included in this branch.

To run the default experiment from the repository folder:

```powershell
python -m voc_sar --config configs/default.yaml --validate-only
python -m voc_sar --config configs/default.yaml
```

Each run writes its configuration, episode traces, diagnostic contexts, summary, and charts to a new timestamped folder under `results/`.

## Repository map

- [`report_final.pdf`](report_final.pdf) — final course report.
- [`demo/`](demo/) — interactive explanation and playable simulator.
- [`src/voc_sar/`](src/voc_sar/) — grid environment, role policies, communication selection, value estimates, and experiment runner.
- [`configs/`](configs/) — default and research experiment settings.
- [`results/`](results/) — saved runs and analyses used by the report.
- [`tests/`](tests/) — automated checks for the simulator and research experiment code.

## Scope

This is a controlled simulator study with fixed task policies. The communication decision starts from an existing structured report; the project does not generate natural-language messages or model a radio network. See the report for the experiment design, findings, and limitations.

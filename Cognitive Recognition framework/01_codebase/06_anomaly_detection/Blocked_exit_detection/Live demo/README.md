# Live Demo Dashboard

Run from the repository root:

```powershell
pip install -r "01_codebase\06_anomaly_detection\Blocked_exit_detection\Live demo\requirements.txt"
streamlit run "01_codebase\06_anomaly_detection\Blocked_exit_detection\Live demo\app_demo.py"
```

The dashboard discovers the newest completed Hallway 1 run under
`04_outputs_runs_and_logs/conference_demo`, loads its semantic-map SQLite data,
and provides a radius slider with collision assessment. Set `RERUN_VIEWER_URL`
to embed a separately hosted Rerun Web viewer in the right panel. The current
Rerun SDK in this environment does not expose `serve_web`, so the app displays a
local Plotly 3D fallback until a Web viewer endpoint is available.

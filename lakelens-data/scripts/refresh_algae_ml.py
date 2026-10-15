
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Refresh inputs and regenerate predictions.
# Do not retrain the model during each refresh.
scripts = [
    "export_algae_history.py",
    "build_algae_dataset.py",
    "fetch_historical_weather.py",
    "build_algae_weather_dataset.py",
    "predict_algae.py",
]

for script in scripts:
    print(f"\n===== Running {script} =====", flush=True)

    subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts" / script)],
        cwd=ROOT,
        check=True
    )

print("\nAlgae ML refresh completed successfully.")

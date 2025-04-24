import os
import subprocess
from dashboard.app import main

if __name__ == "__main__":
    main()


print("Starting Football Analytics Pipeline...")

scripts = [
    "fetch_statsbomb.py",
    "fetch_shots_data.py",
    "fetch_passing_data.py",
    "preprocess_xG.py",
    "train_xG_model.py"
]

for script in scripts:
    print(f"▶ Running {script}...")
    result = subprocess.run(["python", os.path.join("src", script)])
    if result.returncode != 0:
        print(f"Error while running {script}. Check logs.")
        break
else:
    print("Data pipeline completed successfully!")
    print("▶ You can now run the dashboard with:")
    print("   streamlit run src/dashboard.py")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import timedelta

# Define the task information
tasks = [
    {"Task": "Phase 1: Planning and Setup", "Duration": 2, "Start Week": 1, "End Week": 2},
    {"Task": "Phase 2: Data Collection and Preprocessing", "Duration": 3, "Start Week": 3, "End Week": 5},
    {"Task": "Phase 3: Model Selection and Training", "Duration": 4, "Start Week": 6, "End Week": 9},
    {"Task": "Phase 4: Disease Spot Detection and Health Scoring", "Duration": 3, "Start Week": 10, "End Week": 12},
    {"Task": "Phase 5: Evaluation and Testing", "Duration": 2, "Start Week": 13, "End Week": 14},
    {"Task": "Phase 6: Deployment and Documentation", "Duration": 2, "Start Week": 15, "End Week": 16},
]

# Convert data into a DataFrame
df = pd.DataFrame(tasks)

# Define the start and end date of the project
project_start = pd.to_datetime("2024-07-25")  # Example start date for Week 1
df["Start Date"] = project_start + pd.to_timedelta((df["Start Week"] - 1) * 7, unit="D")
df["End Date"] = project_start + pd.to_timedelta((df["End Week"] - 1) * 7, unit="D")

# Create the Gantt chart
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each task as a bar
for i, task in df.iterrows():
    ax.barh(task["Task"], (task["End Date"] - task["Start Date"]).days, left=task["Start Date"], color="skyblue", edgecolor="black")

# Set labels and title
ax.set_xlabel("Timeline")
ax.set_ylabel("Project Phases")
ax.set_title("Gantt Chart")

# Format the x-axis to show dates
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

# Rotate date labels for readability
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
plt.savefig("gantt_chart/gantt_chart.png")

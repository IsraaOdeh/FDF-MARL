import pandas as pd
import ast

# Path to your CSV file
csv_file = "TR-adults-keras_testing_DI - base.csv"

# Read the CSV file (assuming only one column with info dicts)
df = pd.read_csv(csv_file, header=None, names=["reward","f1","fairness","-","avg","sc-info","info"])

# Container for all rows
records = []
df = df[df.index % 2 != 0]

for row in df["info"]:
    # Safely evaluate the string into a dictionary
    data = ast.literal_eval(row)

    # fairness_data = data["agentsFairness"]
    DI = data["DI"]
    f1_data = data["agentsF1"]

    # For each agent in the fairness dict (clients and server)
    for agent, DI_dict in DI.items():

        entry = {
            "Agent": agent,
            "F1": f1_data.get(agent, None)
        }

        for group_id in range(1, 7):
            if DI_dict == 0:
                entry[f"Group{group_id}"] = 0
            else:
                entry[f"Group{group_id}"] = DI_dict.get(group_id, None)

        records.append(entry)

# Convert to DataFrame
result_df = pd.DataFrame(records)

# Save to CSV
result_df.to_csv("results-formatted-DI (base).csv", index=False)

# Print the result
print(result_df)

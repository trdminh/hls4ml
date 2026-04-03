import json
import pandas as pd
import os

base_folder = "fan-v3-export"
output_folder = "data"

import json
import pandas as pd
import os

label_file = "fan-v3-export/info.labels"

os.makedirs(output_folder, exist_ok=True)


with open(label_file, "r", encoding="utf-8") as f:
    label_data = json.load(f)

label_dict = {}

for item in label_data["files"]:
    path = item["path"]
    label = item["label"]["label"]

    json_path = os.path.join(base_folder, path)

    if not os.path.exists(json_path):
        print(f"Không tìm thấy: {json_path}")
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    columns = [s["name"] for s in data["payload"]["sensors"]]
    values = data["payload"]["values"]

    df = pd.DataFrame(values, columns=columns)

    df["label"] = label

    if label not in label_dict:
        label_dict[label] = []

    label_dict[label].append(df)

for label, dfs in label_dict.items():
    final_df = pd.concat(dfs, ignore_index=True)
    output_path = os.path.join(output_folder, f"{label}.csv")
    final_df.to_csv(output_path, index=False)

    print(f"Đã lưu: {output_path}")
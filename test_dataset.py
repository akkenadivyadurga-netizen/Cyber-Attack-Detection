import pandas as pd

file_path = "MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv"

df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nLabel distribution:")
print(df["Label"].value_counts())
venv) PS C:\Users\mdsat\OneDrive\Cyber_Attack_Detection_Project> python train_model.py
Dataset loaded!
Shape: (100000, 79)

Labels:
Label
0    96874
1     3126
Name: count, dtype: int64
(venv) PS C:\Users\mdsat\OneDrive\Cyber_Attack_Detection_Project> 

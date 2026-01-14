import pandas as pd
import config

# Wczytaj tak samo jak w treningu
df = pd.read_csv(config.CSV_FILE)

print("PIERWSZE 5 WIERSZY:")
print(df.head())

print("\nCZY KOLUMNY SĄ DOBRZE NAZWANE?")
print(df.columns)

print("\nPRZYKŁAD TRENINGOWY:")
# Symulacja tego co wchodzi do modelu
row = df.iloc[0]
input_text = config.MODEL_PREFIX + str(row[0]) # Zakładam, że angielski jest w 1 kolumnie
target_text = str(row[1]) # Zakładam, że elficki jest w 2 kolumnie

print(f"INPUT:  '{input_text}'")
print(f"TARGET: '{target_text}'")
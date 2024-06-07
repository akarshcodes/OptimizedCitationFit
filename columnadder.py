import pandas as pd

file_path = '/mnt/data/Adcock-2010.csv'
df = pd.read_csv(file_path)

df['Serial Number'] = range(1, len(df) + 1)

output_file_path = '/mnt/data/Adcock-2010_with_serial_numbers.csv'
df.to_csv(output_file_path, index=False)

output_file_path

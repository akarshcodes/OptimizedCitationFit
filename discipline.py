# -*- coding: utf-8 -*-
"""Discipline.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FN6s0i-bbl-XxWvmC4opMSFjydaiNq_G
"""

!pip install xlsxwriter pandas openpyxl numpy

import pandas as pd
from google.colab import files

disciplines = ["Archi", "Law", "Anthro", "Chem", "Business", "Med", "Psych"]
entries = []

for discipline in disciplines:
    for i in range(1, 33):
        entries.append(f"{discipline}{i}")

df = pd.DataFrame({'Discipline': entries})

output_file = "/content/Disciplines.xlsx"
df.to_excel(output_file, index=False)

files.download(output_file)
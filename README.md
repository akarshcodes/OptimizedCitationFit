
# OptimizedCitationFit

OptimizedCitationFit is a Python-based model designed for optimizing citation data fitting. The tool allows users to upload Excel files, perform parameter optimization, visualize results, and generate comprehensive reports. The package leverages popular libraries such as Pandas, NumPy, SciPy, Matplotlib, and python-docx to streamline citation data analysis and modeling.

## Features
- **Excel Uploads:** Easily upload citation data in .xlsx format.
- **Parameter Optimization:** Utilize advanced algorithms to optimize fitting parameters for citation data.
- **Visualization:** Generate detailed plots and visualizations to analyze citation trends.
- **Report Generation:** Automatically generate professional reports in Word format summarizing the analysis.

## Prerequisites
- Python 3.x
- Required Python libraries: Pandas, NumPy, SciPy, Matplotlib, python-docx

Install the required libraries using pip:
```bash
pip install pandas numpy scipy matplotlib python-docx
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/akarshcodes/OptimizedCitationFit.git
   ```
2. Navigate to the project directory:
   ```bash
   cd OptimizedCitationFit
   ```

## Usage
1. **Upload .xlsx Data:** Place your citation data in an Excel file.
2. **Run the Scripts:** Execute the provided Python scripts to perform data fitting, optimization, and visualization.
   - Example:
     ```bash
     python h_star.py
     ```
3. **View Results:** Check the output directory for visualizations and the generated Word report.

## Code Structure
- **h_star.py:** Calculates h-index related metrics.
- **eq_calculator.py:** Equation-based calculations for citation analysis.
- **plotsFINAL.ipynb:** Notebook for generating final plots.
- **final_file_handling.ipynb:** Handles file processing and data management.
- **scatter_for_2_files.py:** Creates scatter plots comparing two datasets. 
- ** Similarly refer other python files/ jupyter notebooks:** 

## Examples
Refer to the provided Jupyter Notebooks for step-by-step examples:
- **approximation2.ipynb:** Detailed approximation methods.
- **regression_checker_individual.ipynb:** Check regression accuracy for individual data sets.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For any queries on naming convention, please open an issue in the repository or contact the maintainers. (Akarsh Srivastava or Prof. YC Tay, Dept. of Computer Science, NUS Singapore)

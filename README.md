## BMI706 2025 Final Project: Breast Cancer Clinical Trial Visualization 

*Group member: Alice Zhang, Junyang Deng, Cece Jen, Jason Liang*

**Link to app: https://bmi706projectapppy-ikpyb3vhcaphqqk7wdawyh.streamlit.app/**

This project presents an interactive visualization tool built using Altair and Streamlit to explore global patterns in breast cancer clinical trials. The dashboard enables users to investigate how breast cancer research activity has evolved over time, across intervention types, across sponsor groups, and across global regions.

### Exploratory Analysis Goals

The purpose of our exploratory visualization tool is to understand how breast cancer clinical research has evolved globally from 1977 to 2025. Through an interactive, multidimensional dashboard, we aim to uncover long-term temporal trends in trial initiation and completion, characterize shifts in the therapeutic landscape, assess how different sponsor types contribute to research activity, and examine geographic patterns to identify high-activity research hubs and underrepresented regions. By integrating these dimensions, the visualization supports hypothesis generation for clinical researchers and provides a comprehensive view of how the field has evolved and been shaped by diverse scientific, institutional, and geographic factors.

### Task

1) Temporal trends in breast cancer clinical research activity 
2) Geographic concentration and distribution of breast cancer clinical trials
3) Distribution and evolution of intervention types in breast cancer trials
4) Influence of sponsor type across study period, country, and study status

### Data

All data used for this visualization is stored in the `data/` folder. 

- `breast_cancer_raw.csv`: initial dataset extracted before cleaning and processing
- `breast_cancer.csv`: processed dataset used in the dashboard
- `world_coord.csv`: latitude/longitude coordinates used to generate the proportional-symbol world map

### User guide

`BMI706_final_project.pdf` provides a comprehensive walkthrough of the visualization tools, including instructions on how to navigate the dashboard, apply filters, interpret each chart, and explore trends in breast cancer clinical trials.

###  References

- Clinical Trials Transformation Initiative. AACT Database. https://aact.ctti-clinicaltrials.org/
- U.S. National Library of Medicine. ClinicalTrials.gov. https://clinicaltrials.gov/
- National Cancer Institute. Breast Cancer Treatment (PDQ). https://www.cancer.gov/types/breast/patient/breast-treatment-pdq


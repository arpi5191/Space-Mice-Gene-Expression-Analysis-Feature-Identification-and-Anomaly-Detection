# Space Mice Gene Expression Analysis: Feature Identification and Anomaly Detection

Implements Principal Component Analysis (PCA) on a NASA dataset to identify the features contributing most significantly to the mice's gene expression. Utilizes a machine learning pipeline with the Python package, Ibis, to perform the analysis. Additionally, applies the Isolation Forest algorithm to detect anomalies in the dataset, uncovering potential outliers.

# Instructions:
  1) Download the metadata.csv and data.csv files from the NASA website at https://osdr.nasa.gov/bio/repo/data/studies/OSD-665 to a platform like Jupyter Notebook or Google Colab.
  2) Download the PCA.ipynb notebook and upload it to the same platform. Open the PCA.ipynb notebook and run it to perform Principal Component Analysis (PCA) on the dataset. The analysis will identify the most significant factors—condition, library preparation (libprep), and strain—that contribute to gene expression variation between mice in space and on the ground.
  3) Download the Ibis_PCA.py script and upload it to the same platform. Run "python Ibis_PCA.py" to perform PCA analysis through a machine learning pipeline utilizing the Ibis package.
  4) Download the IsolationForestAlgorithm.ipynb notebook and upload it to the same platform. Open the IsolationForestAlgorithm.ipynb notebook and run it to apply the Isolation Forest Algorithm to the dataset. This analysis will identify anomalies in the dataset.

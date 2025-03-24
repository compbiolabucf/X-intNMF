python X-intNMF-run.py \
--omics_input \
./sample_processed_data/BRCA_2omics/mRNA.parquet \
./sample_processed_data/BRCA_2omics/miRNA.parquet \
--interaction_input \
./sample_processed_data/BRCA_2omics/interaction_mRNA_miRNA.parquet \
--gpu 0 \
--mlflow_uri http://localhost:6969 \
--mlflow_experiment_name test_experiment
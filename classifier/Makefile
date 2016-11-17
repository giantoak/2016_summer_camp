# Local data used to build data frames
HOMOLOGY_DB = data/initial/homology_160722.db
LATTICE_JSONS = data/initial/stripped_before_201605
ID_FILE = data/initial/cdr_ids_to_get.txt

# Target for setting up data/initial
setup:
	mkdir data
	mkdir data/generated
	mkdir data/classifier
	mkdir data/classifier/work/
	mkdir data/initial
	mkdir data/initial/stripped_before_201605

	aws s3 cp s3://giantoak.memex/2016_summer_camp/cr_ids_to_get.txt data/initial
	aws s3 cp s3://giantoak.memex/2016_summer_camp/homology_160722.db.gz data/initial
	cd data/initial
	gunzip homology_160722.db.gz
	cd ../..

	aws s3 cp --recursive s3://giantoak.memex/2016_summer_camp/stripped_before_201605/ data/initial/stripped_before_201605

data/classifier/work/true_positives_text.json: parse_cp.py data/initial/CP1_train_ads.json
	python parse_cp.py
# Targets for building initial data
data/generated/giantoak_df.pkl: \
	$(ID_FILE)
	ipython make_giantoak_df.py $(ID_FILE)

data/generated/lattice_df.pkl: $(ID_FILE) make_lattice_df.py
	ipython make_lattice_df.py $(ID_FILE) $(LATTICE_JSONS)

data/generated/homology_df.pkl: $(ID_FILE) $(HOMOLOGY_DB)
	ipython make_homology_df.py $(ID_FILE) $(HOMOLOGY_DB)

data/generated/image_df.pkl: \
	$(ID_FILE)
	ipython make_image_df.py $(ID_FILE)

pre_merge: \
	data/generated/lattice_df.pkl \
	data/generated/homology_df.pkl \
	data/generated/giantoak_df.pkl \
	data/generated/image_df.pkl 


### Work on new munge of text only plus stuff pulled from HBASE
negative_sample_full_docs_v2.jl: negative_sample_cdr_id_and_phone_1.csv fetch_true_and_false_samples.py
	# Note as written this takes WEEKS to pull 120K ads.
	ipython fetch_true_and_false_samples.py
data/initial/CP1_train_ads.json:
	aws s3 cp s3://giantoak.memex/models/CP1_train_ads.json.gz data/initial/
	rm -f data/initial/CP1_train_ads.json
	gunzip data/initial/CP1_train_ads.json.gz
cdr_id_and_phones_from_cdr.txt: get_cdr_ids.py
	ipython get_cdr_ids.py
negative_sample_cdr_id_and_phone_1.csv: cdr_id_and_phones_from_cdr.txt draw_negative_sample.py
	ipython draw_negative_sample.py
data/initial/cp1_evaluation_data.json: 
	aws s3 cp s3://giantoak.memex/models/cp1_evaluation_data.json.gz data/initial/
	rm -f data/initial/cp1_evaluation_data.json
	gunzip data/initial/cp1_evaluation_data.json.gz

giantoak_ht_model.pkl: negative_sample_full_docs_v2.jl data/initial/CP1_train_ads.json data/initial/cp1_evaluation_data.json models/price_imputation_text_extractor.pkl models/age_imputation_text_extractor.pkl
	ipython model_munge.py

models/price_imputation_text_extractor.pkl: 
	mkdir -p models/
	aws s3 cp s3://giantoak.memex/2016_summer_camp/price_imputation_text_extractor.pkl models/
	aws s3 cp s3://giantoak.memex/2016_summer_camp/price_imputation_model.pkl models/

models/age_imputation_text_extractor.pkl:
	mkdir -p models/
	aws s3 cp s3://giantoak.memex/2016_summer_camp/age_imputation_text_extractor.pkl models/
	aws s3 cp s3://giantoak.memex/2016_summer_camp/age_imputation_model.pkl models/

### End work on new munge
data/generated/negative_sample.csv: sample_cdr_ids.py
	ipython sample_cdr_ids.py

# Targets for merging initial data
data/generated/merged_df.pkl: \
	pre_merge
	ipython make_merged_df.py

data/generated/merged_phone_df.pkl: \
	data/generated/merged_df.pkl
	ipython group_data_by_phone.py

merged: \
	data/generated/merged_df.pkl \
	data/generated/merged_phone_df.pkl


# Target for bulding a classifier in the notebook
classifier: \
	data/generated/merged_phone_df.pkl
	runipy -o Classifier.ipynb

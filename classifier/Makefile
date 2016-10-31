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

data/initial/CP1_train_ads.json:
	aws s3 cp s3://giantoakmemex/CP1_train_ads.json data/initial/
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

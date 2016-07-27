# Local data used to build data frames
HOMOLOGY_DB = data/initial/homology_160722.db
LATTICE_JSONS = data/initial/stripped_before_201605
ID_FILE = data/initial/cdr_ids_to_get.txt


# Targets for building initial data
data/generated/giantoak_df.pkl: \
	$(ID_FILE)
	ipython make_giantoak_df.py $(ID_FILE)

data/generated/lattice_df.pkl: \
	$(ID_FILE) \
	$(LATTICE_JSONS)
	ipython make_lattice_df.py $(ID_FILE) $(LATTICE_JSONS)

data/generated/homology_df.pkl: \
	$(ID_FILE) \
	HOMOLOGY_DB
	ipython make_homology_df.py $(ID_FILE) $(HOMOLOGY_DB)

data/generated/image_df.pkl: \
	$(ID_FILE)
	ipython make_image_df.py $(ID_FILE)

pre_merge: \
	data/generated/lattice_df.pkl \
	data/generated/homology_df.pkl \
	data/generated/giantoak_df.pkl \
	data/generated/image_df.pkl \


# Targets for merging initial data
data/generated/merged_df.pkl: \
	pre_merge
	ipython make_merged_data.py

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

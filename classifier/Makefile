HOMOLOGY_DB = /Users/pmlandwehr/wkbnch/memex/memex_queries/homology_160722.db
LATTICE_JSONS = /Users/pmlandwehr/wkbnch/memex/memex_ad_features/lattice_data/stripped_before_201605

## These targets build merged data sets

giantoak_df.pkl:
	ipython make_giantoak_df.py

lattice_df.pkl:
	ipython make_lattice_df.py LATTICE_JSONS

homology_df.pkl: lattice_df.pkl
	ipython make_homology_df.py HOMOLOGY_DB

image_df.pkl: lattice_df.pkl
	ipython make_image_df.py

merged_df.pkl: lattice_df.pkl image_df.pkl homology_df.pkl
	ipython make_merged_data.py

merged_phone_df.pkl: merged_df.pkl
	ipython group_data_by_phone.py

## These targets build the classifier

classifier: merged_phone_df.pkl
	runipy Sixth\ Classifer.ipynb

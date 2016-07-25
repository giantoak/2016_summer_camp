## These targets build merged data sets

giantoak_df.pkl:
	ipython make_giantoak_df.py

lattice_df.pkl:
  ipython make_lattice_df.py

homology_df.pkl: lattice_data.pkl
	ipython make_homology_df.py

image_df.pkl: lattice_data.pkl
	ipython make_image_df.py

merged_df.pkl: lattice_data.pk image_data.pkl homology_data.pkl
	ipython make_merged_data.py

merged_phone_df.pkl: merged_df.pkl
	ipython group_data_by_phone.py

## These targets build the classifier

classifier: merged_phone_df.pkl
	runipy Sixth\ Classifer.ipynb

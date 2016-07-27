## Data you need to run the classifier (and how to get it)

**BEFORE ANYTHING ELSE**:
```bash
mkdir data
mkdir data/generated
mkdir data/initial
```

### `data/initial/cdr_ids_to_get.txt`
This could be any list of CDR `_id`s that you wanted to provide. To get a set made from the CP 1 training data and Kyle Hundeman's unchecked sample for JPL, use:
```bash
aws s3 cp s3://giantoak.memex/2016_summer_camp/cr_ids_to_get.txt data/initial
```

### `data/initial/homology_160722.db`
```bash
aws s3 cp s3://giantoak.memex/2016_summer_camp/homology_160722.db.gz data/initial
cd data/initial
gunzip homology_160722.db.gz
```

### `data/initial/stripped_before_201605/*.json.gz`
(This will be going away, darn it, but for the moment we need it)
```bash
aws s3 cp --recursive s3://giantoak.memex/2016_summer_camp/stripped_before_201605/ data/initial
```

## Running The Classifier
No guarantees, but this should do the job:
```bash
make pre_merge   # Make the input data frames
make merge       # Merge the data frames
make classifier  # Train the classifier on the merged data
```

## Basic To-Do List
To be more official these should go in JIRA or the github issue tracker, but this will do fine.
- [ ] Get Lattice data in Hbase (raw)
  - [ ] Unfancy (ID: JSON)
  - [ ] Fancy (ID: all dict keys)
- [ ] Get image DF working.
- [ ] Test with sample data.

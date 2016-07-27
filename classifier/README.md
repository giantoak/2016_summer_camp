# Data you need (and how to get it)

**BEFORE ANYTHING ELSE**:
```bash
mkdir data
mkdir data/generated
mkdir data/initial
```

## `data/initial/cdr_ids_to_get.txt`
This could be any list of CDR `_id`s that you wanted to provide. To get a set made from the CP 1 training data and Kyle Hundeman's unchecked sample for JPL, use:
```bash
aws s3 cp s3://giantoak.memex/2016_summer_camp/cr_ids_to_get.txt data/initial
```

## `data/initial/homology_160722.db`
```bash
aws s3 cp s3://giantoak.memex/2016_summer_camp/homology_160722.db.gz data/initial
cd data/initial
gunzip homology_160722.db.gz
```

## `data/initial/stripped_before_201605/*.json.gz`
(This will be going away, darn it, but for the moment we need it)
```bash
aws s3 cp --recursive s3://giantoak.memex/2016_summer_camp/stripped_before_201605/ data/initial
```

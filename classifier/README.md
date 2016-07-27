## Setting up
Assuming you have the Giant Oak AWS Creds, setup should be:

```bash
make setup
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

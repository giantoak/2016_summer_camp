"""
Reads Giant Oak data produced by Jeff / Gabriel
"""

def main():
    # This should replace our use of the lattice_df.
    import pandas as pd

    df = pd.DataFrame({})
    df.to_pickle('data/generated/giantoak_df.pkl')

if __name__ == "__main__":
    main()

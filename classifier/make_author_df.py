"""
Reads MIT-LL Author data. Check with Charli / Joe / (Kelly & other week 3 person?)
"""

def main():
    import pandas as pd

    df = pd.DataFrame({})
    df.to_pickle('data/generated/author_df.pkl')

if __name__ == "__main__":
    main()

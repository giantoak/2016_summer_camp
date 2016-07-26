"""
This is still stolen from the last QPR! We need to update it.
"""


def main():
    from helpers import dummify_df

    df = pd.read_pickle('data/generated/merged_df.pkl')

    numerical_vars = ['age',
                      'price',
                      'duration_in_mins',
                      'price_per_min',
                      'images_count',
                      'exp_ads_from_simimages_count',
                      'similar_images_count']

    phone_level_vars = ['n_ads',
                        'n_distinct_locations',
                        'location_tree_length',
                        'n_outcall',
                        'n_incall',
                        'n_incall_and_outcall',
                        'n_cooccurring_phones']

    missing_vars = ['missing_{}'.format(col) for col in numerical_vars]

    # Missing images means 0 images
    # We've solved that with fillna(0)
    missing_vars.remove('missing_images_count')

    for col in missing_vars:
        df[col] = ~df[col[len('missing_'):]].notnull().astype(int)

    numerical_df = df.groupby(
        'phone')[numerical_vars + missing_vars].describe().unstack()
    print(numerical_df.shape)
    numerical_df = numerical_df.dropna(0, 'all')
    print(numerical_df.shape)

    phone_level_df = df.groupby('phone')[phone_level_vars].max()
    print(phone_level_df.shape)
    phone_level_df = phone_level_df.dropna(0, 'all')
    print(phone_level_df.shape)

    flag_dummies = dummify_df(df.loc[:, ['phone', 'flag', 'ethnicity']],
                              ['flag', 'ethnicity'],
                              '|')
    discrete_df = flag_dummies.groupby('phone').mean()

    phone_level_df = phone_level_df.join(
        [numerical_df, discrete_df], how='outer')
    # phone_level_df['has_images'] = df.groupby('phone')['has_images'].max()
    phone_level_df['class'] = df.groupby('phone')['class'].max()

    phone_level_df = phone_level_df.fillna(-1).reset_index()
    phone_level_df['phone'] = phone_level_df['index']
    del phone_level_df['index']

    print(phone_level_df.shape)

    phone_level_df.columns = [x if not isinstance(
        x, tuple) else ':'.join(x) for x in phone_level_df.columns]

    phone_level_df.to_pickle('data/generated/phone_level_merged_df.pkl')

if __name__ == "__main__":
    pass

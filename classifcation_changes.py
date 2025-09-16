import pandas as pd
from download_dataset import download_dataset

# List of religion and ethnicity columns to combine
RELIGION_COLS = [
    'target_christian_bool', 'target_muslim_bool', 'target_jewish_bool',
    'target_hindu_bool', 'target_buddhist_bool', 'target_sikh_bool',
    'target_other_religion_bool'
]
RELIGION_LABELS = [
    'Christian', 'Muslim', 'Jewish', 'Hindu', 'Buddhist', 'Sikh', 'Other'
]

ETHNICITY_COLS = [
    'target_white_bool', 'target_black_bool', 'target_asian_bool',
    'target_latino_bool', 'target_indigenous_bool', 'target_other_ethnicity_bool'
]
ETHNICITY_LABELS = [
    'White', 'Black', 'Asian', 'Latino', 'Indigenous', 'Other'
]

NATIONALITY_COLS = [
    'target_american_bool', 'target_european_bool', 'target_african_bool',
    'target_asian_nationality_bool', 'target_other_nationality_bool'
]
NATIONALITY_LABELS = [
    'American', 'European', 'African', 'Asian', 'Other'
]

GENDER_COLS = [
    'target_male_bool', 'target_female_bool', 'target_nonbinary_bool',
    'target_lgbtq_bool', 'target_other_gender_bool'
]
GENDER_LABELS = [
    'Male', 'Female', 'Nonbinary', 'LGBTQ', 'Other'
]

DISABILITY_COLS = [
    'target_disabled_bool', 'target_abled_bool', 'target_other_disability_bool'
]
DISABILITY_LABELS = [
    'Disabled', 'Abled', 'Other'
]

def combine_columns(df):
    # Target group merges
    def merge_group(df, prefix, options, new_col):
        df[new_col] = None
        for opt in options:
            col = f'{prefix}_{opt}'
            if col in df.columns:
                df.loc[df[col] == 1, new_col] = opt.replace('_', ' ').title()
        drop_cols = [f'{prefix}_{opt}' for opt in options if f'{prefix}_{opt}' in df.columns]
        df = df.drop(columns=drop_cols)
        return df

    # Target merges
    df = merge_group(df, 'target_race', ['asian','black','latinx','middle_eastern','native_american','pacific_islander','white','other'], 'target_race')
    df = merge_group(df, 'target_religion', ['atheist','buddhist','christian','hindu','jewish','mormon','muslim','other'], 'target_religion')
    df = merge_group(df, 'target_origin', ['immigrant','migrant_worker','specific_country','undocumented','other'], 'target_origin')
    df = merge_group(df, 'target_gender', ['men','non_binary','transgender_men','transgender_unspecified','transgender_women','women','other'], 'target_gender')
    df = merge_group(df, 'target_sexuality', ['bisexual','gay','lesbian','straight','other'], 'target_sexuality')
    df = merge_group(df, 'target_age', ['children','teenagers','young_adults','middle_aged','seniors','other'], 'target_age')
    df = merge_group(df, 'target_disability', ['physical','cognitive','neurological','visually_impaired','hearing_impaired','unspecific','other'], 'target_disability')

    # Annotator merges
    df = merge_group(df, 'annotator_race', ['asian','black','latinx','middle_eastern','native_american','pacific_islander','white','other'], 'annotator_race')
    df = merge_group(df, 'annotator_religion', ['atheist','buddhist','christian','hindu','jewish','mormon','muslim','nothing','other'], 'annotator_religion')
    df = merge_group(df, 'annotator_sexuality', ['bisexual','gay','straight','other'], 'annotator_sexuality')
    df = merge_group(df, 'annotator_ideology', ['extremeley_conservative','conservative','slightly_conservative','neutral','slightly_liberal','liberal','extremeley_liberal','no_opinion'], 'annotator_ideology')
    df = merge_group(df, 'annotator_income', ['<10k','10k-50k','50k-100k','100k-200k','>200k'], 'annotator_income')
    df = merge_group(df, 'annotator_gender', ['men','women','non_binary','prefer_not_to_say','self_describe'], 'annotator_gender')

    return df

def main():
    dataset = download_dataset('default')
    if dataset is None:
        print('Dataset download failed.')
        return
    df = dataset['train'].to_pandas()
    print('All columns in the original dataset:')
    for col in df.columns:
        print(col)
    df_new = combine_columns(df)
    print('Columns combined:')
    print('Religion columns:', RELIGION_COLS, '-> religion')
    print('Ethnicity columns:', ETHNICITY_COLS, '-> ethnicity')
    print('Nationality columns:', NATIONALITY_COLS, '-> nationality')
    print('Gender/Sexuality columns:', GENDER_COLS, '-> gender_sexuality')
    print('Disability columns:', DISABILITY_COLS, '-> disability')
    print('\nOriginal dataset shape:', df.shape)
    print('New dataset shape:', df_new.shape)
    print('\nFirst 5 rows of the new dataset:')
    print(df_new.head())
    print('\nNew columns:', df_new.columns.tolist())
    print('\nAll columns in the new dataset:')
    for col in df_new.columns:
        print(col)

if __name__ == "__main__":
    main()

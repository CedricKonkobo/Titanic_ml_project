
# import pandas as pd
# import numpy as np 
# def create_features(df):
#         """Create new features from raw data."""
#         df = df.copy()
        
#         # Family size
#         df['familySize'] = df['sibsp'] + df['parch'] + 1
#         df['isAlone'] = (df['familySize'] == 1).astype(int)
        
#         # Title extraction
#         df['title'] = df['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
#         df['title'] = df['title'].replace(['Lady', 'Countess', 'Capt', 'Col', 
#                                            'Don', 'Dr', 'Major', 'Rev', 'Sir', 
#                                            'Jonkheer', 'Dona'], 'Rare')
#         df['title'] = df['title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
        
#         # Age categories
#         df['ageBand'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100], 
#                                labels=['Child', 'Teen', 'Adult', 'Senior', 'Elder'])
        
#         # Fare categories
#         df['fareBand'] = pd.qcut(df['fare'], 4, labels=['Low', 'Med', 'High', 'Premium'], duplicates='drop')
        
#         # Has Cabin
#         df['hasCabin'] = df['cabin'].notna().astype(int)
        
#         # Deck from Cabin
#         df['deck'] = df['cabin'].str[0] if 'cabin' in df.columns else 'U'
#         df['deck'] = df['deck'].fillna('U')
        
#         return df

# from src.data.load_data import load_raw_data
    
# train, _ = load_raw_data()
# df_train = create_features(train)
# print(df_train.head())
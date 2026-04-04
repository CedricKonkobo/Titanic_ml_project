import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

class FeatureEngineer:
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None
        
    def create_features(self, df):
        """Create new features from raw data."""
        df = df.copy()
        
        # Family size
        df['familySize'] = df['sibsp'] + df['parch'] + 1
        df['isAlone'] = (df['familySize'] == 1).astype(int)
        
        # Title extraction
        df['title'] = df['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df['title'] = df['title'].replace(['Lady', 'Countess', 'Capt', 'Col', 
                                           'Don', 'Dr', 'Major', 'Rev', 'Sir', 
                                           'Jonkheer', 'Dona'], 'Rare')
        df['title'] = df['title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
        
        # Age categories
        df['ageBand'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100], 
                               labels=['Child', 'Teen', 'Adult', 'Senior', 'Elder'])
        
        # Fare categories
        df['fareBand'] = pd.qcut(df['fare'], 4, labels=['Low', 'Med', 'High', 'Premium'], duplicates='drop')
        
        # Has Cabin
        df['hasCabin'] = df['cabin'].notna().astype(int)
        
        # Deck from Cabin
        df['deck'] = df['cabin'].str[0] if 'cabin' in df.columns else 'U'
        df['deck'] = df['deck'].fillna('U')
        
        return df
    
    def handle_missing(self, df):
        """Handle missing values intelligently."""
        df = df.copy()
        
        # Age: impute by Title and Pclass median
        df['age'] = df.groupby(['title', 'pclass'])['age'].transform(
            lambda x: x.fillna(x.median()) if not x.median() == np.nan else x.fillna(df['age'].median())
        )
        
        # Embarked: mode
        df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
        
        # Fare: median
        df['fare'] = df['fare'].fillna(df['fare'].median())
        
        return df
    
    def build_preprocessor(self, X):
        """Build sklearn preprocessor."""
        numeric_features = ['age', 'fare', 'familySize', 'sibsp', 'parch']
        categorical_features = ['pclass', 'sex', 'embarked', 'title', 'ageBand', 'isAlone', 'hasCabin']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        return self.preprocessor.fit(X)
    
    def fit_transform(self, df, fit=True):
        """Full pipeline: engineer + preprocess."""
        df = self.create_features(df)
        df = self.handle_missing(df)
        
        X = df.drop(['survived', 'passenger_id', 'name', 'ticket', 'cabin'], axis=1, errors='ignore')
        y = df['survived'] if 'survived' in df.columns else None
        
        if fit:
            self.build_preprocessor(X)
            
        X_processed = self.preprocessor.transform(X)
        
        # Get feature names
        feature_names = (self.preprocessor.named_transformers_['cat']
                        .named_steps['onehot']
                        .get_feature_names_out(['pclass', 'sex', 'embarked', 'title', 'ageBand', 'isAlone', 'hasCabin']))
        self.feature_names = list(self.preprocessor.transformers_[0][2]) + list(feature_names)
        
        return X_processed, y, df

if __name__ == "__main__":
    from src.data.load_data import load_raw_data
    
    train, _ = load_raw_data()
    fe = FeatureEngineer()
    X, y, df_enhanced = fe.fit_transform(train)
    
    print(f"Features created: {X.shape[1]}")
    print(f"Top features: {fe.feature_names[:5]}")
    
    # Save preprocessor for API
    joblib.dump(fe.preprocessor, "models/preprocessor.joblib")
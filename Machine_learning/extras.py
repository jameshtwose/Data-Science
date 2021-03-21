import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def variance_threshold(df:pd.DataFrame, 
                       threshold: float):
    scaled_df = pd.DataFrame(MinMaxScaler()
             .fit_transform(
                 df.select_dtypes("number")
             ), 
             columns=df.columns)
    summary_df = (scaled_df
     .var()
     .to_frame(name="variance")
     .assign(feature_type=df.dtypes)
     .assign(discard=lambda x: x["variance"] < threshold)
    )
    
    return summary_df


def nunique_threshold(df:pd.DataFrame, 
                       threshold: int):
    scaled_df = pd.DataFrame(MinMaxScaler()
             .fit_transform(
                 df.select_dtypes("number")
             ), 
             columns=df.columns)
    summary_df = (scaled_df
     .apply(lambda x: x.nunique())
     .to_frame(name="nunique")
                  .assign(percent_unique=lambda x: x["nunique"] / df.shape[0] * 100)
     .assign(feature_type=df.dtypes)
     .assign(discard=lambda x: x["nunique"] < threshold)
    )
    
    return summary_df

swell_eda_features_cols = ['MEAN',
 'MAX',
 'MIN',
 'RANGE',
 'KURT',
 'SKEW',
 'MEAN_1ST_GRAD',
 'STD_1ST_GRAD',
 'MEAN_2ND_GRAD',
 'STD_2ND_GRAD',
 'ALSC',
 'INSC',
 'APSC',
 'RMSC',
 'MIN_PEAKS',
 'MAX_PEAKS',
 'STD_PEAKS',
 'MEAN_PEAKS',
 'MIN_ONSET',
 'MAX_ONSET',
 'STD_ONSET',
 'MEAN_ONSET']

swell_eda_target_cols = [
 'condition',
 'Valence',
 'Arousal',
 'Dominance',
 'Stress',
 'MentalEffort',
 'MentalDemand',
 'PhysicalDemand',
 'TemporalDemand',
 'Effort',
 'Performance',
 'Frustration',
 'NasaTLX',
 ]


from preprocess import Preprocessor
import polars as pl

preprocessor = Preprocessor(["../data/GratiMDL.xlsx"],
                            [pl.datetime(2016, 12, 27)])
preprocessor.load_data()
df = preprocessor.preprocess()
print(df.head())
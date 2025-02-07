import pandas as pd
import umap

class Visualizer:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.reducer = umap.UMAP(n_components=self.n_components)

    def fit_transform(self, df, embeddings_col):
        embeddings = df[embeddings_col].tolist()
        reduced_embeddings = self.reducer.fit_transform(embeddings)
        df[f'reduced_{embeddings_col}'] = list(reduced_embeddings)
        return df


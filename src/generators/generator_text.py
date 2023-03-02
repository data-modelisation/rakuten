import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.tools.commons import convert_to_readable_categories
from src.tools.text import pipeline_loader, pipeline_preprocess, pipeline_lang
from src.generators.generator import CommonGenerator


class TextGenerator(CommonGenerator):

    def __init__(self,
                 stem=True,
                 clean=True,
                 translate=True,
                 max_words=100,
                 max_len=100,
                 **kwargs
                 ):

        super().__init__(**kwargs)

        self.translate = translate
        self.stem = stem
        self.clean = clean
        self.max_words = max_words
        self.max_len = max_len

        self.features, self.labels = self.load()

        self.get_counts()
        self.get_imbalanced()
        self.get_correlation()

        if self.translate:
            self.translation()

        self.fit_preprocess()
        self.encode_targets()

    def get_correlation(self,):

        columns = ["words_designation", "length_designation", "words_description", "length_description"]
        df = pd.DataFrame(self.features[:, :4], columns=columns).astype(int)
        
        df["targets"] = self.labels.astype(str)
        
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        
        for column in columns:
            model = ols(f'{column} ~ C(targets)', data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            print(anova_table)

    def get_imbalanced(self,):

        df = pd.DataFrame(self.labels, columns=["prdtypecode"])
        df["prdtypename"] = convert_to_readable_categories(df)

        sns.set(rc={'figure.figsize': (10, 8)})

        graph_category = sns.countplot(
            data=df,
            y="prdtypename",
            order=df.prdtypename.value_counts().index)

        graph_category.set(
            xlabel='Nombre de produits [-]',
            ylabel='Catégories [-]',
            title='Nombre de produits par catégorie (Train)')
        plt.savefig("notebooks/images/imbalanced.png")
        plt.clf()

    def get_counts(self,):

        df = pd.DataFrame(self.features[:, :4], columns=[
                          "words_designation", "length_designation", "words_description", "length_description"])

        fig, axes = plt.subplots(1, 2, figsize=(15, 8))

        graph_desig = sns.histplot(
            data=df,
            kde=True,
            bins=40,
            x="words_designation",
            ax=axes[0])

        graph_descr = sns.histplot(
            data=df,
            kde=True,
            bins=40,
            x="words_description",
            ax=axes[1])

        graph_desig.set(
            xlabel='Nombre de mots [-]',
            ylabel='Total [-]',
            title='Nombre de mots dans `designation` (Train)')

        graph_descr.set(
            xlabel='Nombre de mots [-]',
            ylabel='Total [-]',
            title='Nombre de mots dans `description` (Train)')

        plt.savefig("notebooks/images/words.png")
        plt.clf()

    def translation(self,):

        lang_transformer = pipeline_lang()
        langages = lang_transformer.fit_transform(self.features)

        order_lang = pd.DataFrame(langages, columns=["lang"]).value_counts(normalize=True).sort_values(
            ascending=False).head(25).reset_index().rename(columns={0: "ratio"})

        sns.set(rc={'figure.figsize': (10, 5)})
        graph_lang = sns.barplot(
            data=order_lang,
            x="lang",
            y="ratio",
        )
        graph_lang.set(
            xlabel='Langue [-]',
            ylabel='Nombre de titres / langue [-]',
            title='Analyse de la répartition des langues')

        plt.savefig("notebooks/images/lang.png")
        plt.clf()

    def load(self):

        labels = pd.read_csv(self.csv_labels).prdtypecode
        texts = pd.read_csv(self.csv_texts)

        text_loader = pipeline_loader()
        return text_loader.fit_transform(texts).values, labels.values

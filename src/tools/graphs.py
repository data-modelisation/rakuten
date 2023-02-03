import seaborn as sns


def heatmap(df):
    graph = sns.heatmap(df,fmt='.2f',
         linewidths=2,annot=False)
    return graph

def histplot(df, x, 
    xlabel=None, 
    y_label=None, 
    title=None, 
    ax=[], 
    kde=True):

    graph = sns.histplot(
        data=df,
        kde=True, 
        bins=40,
        x=x,
        ax=ax);

    if xlabel:
        graph.set(xlabel=xlabel)

    if ylabel:
        graph.set(ylabel=ylabel)
        
    if title:
        graph.set(title=title)    

    return graph_desig
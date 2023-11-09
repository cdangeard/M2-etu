import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go

# Title
st.title('Horses Colic Data Set')

# data : https://www.kaggle.com/uciml/horse-colic
# columns : surgery, age, hospital number, rectal temperature, pulse, respiratory rate, temperature of extremities, peripheral pulse, mucous membranes, capillary refill time, pain, peristalsis, abdominal distension, nasogastric tube, nasogastric reflux, nasogastric reflux PH, rectal examination, abdomen, packed cell volume, total protein, abdominocentesis appearance, abdomcentesis total protein, outcome, surgical lesion, type of lesion1, type of lesion2, type of lesion3, cp_data
# types : categorical, categorical, numerical, numerical, numerical, numerical, categorical, categorical, categorical, categorical, categorical, categorical, categorical, categorical, categorical, numerical, categorical, categorical, numerical, numerical, categor
# Load the data from the CSV file
data = pd.read_csv("data./horse.csv")   

# MultiSelect
list_columns = st.multiselect(
    label = "Select columns",
    options = data.columns,
    default = list(data.columns)
    )

# Onglets
dataTab, uniTab, biTab, corTab, abTab, missTab = st.tabs([
    "Horses data",
    "Analyse univariée",
    "Analyse bivariée",
    "Corrélations",
    "Valeurs abérentes",
    "Valeurs manquantes"
    ])

with dataTab:
    st.write('## Data')
    st.write(data[list_columns])


with uniTab:
    # Analyse univariée
    var = st.selectbox(
        "Select column",
        options = data[list_columns].columns
        )
    st.write('## Analyse univariée')
    if var == 'surgical_lesion':
        st.write(data[var].value_counts())
        # Camembert pour la variable à prédire
        fig = px.pie(
            data,
            names=var,
            color_discrete_sequence=["#e41a1c", "#377eb8"]
            )
        st.plotly_chart(fig)
    else:
        # Histogramme pour les autres variables
        fig = px.histogram(
            data,
            x=var, color="surgical_lesion",
            color_discrete_sequence= ["#377eb8", "#e41a1c"]
            )
        st.plotly_chart(fig)

with biTab:
    # Analyse bivariée
    var1 = st.selectbox(
        "Select column 1",
        options = data[list_columns].columns
        )
    var2 = st.selectbox(
        "Select column 2",
        options = data[list_columns].columns.drop(var1)
        )

    #Si deux variables quantitatives : lineChart
    if (data[var1].dtype == 'float64') & (data[var2].dtype == 'float64'):
        st.write('## Analyse bivariée : continue vs continue')
        fig = px.scatter(
            data,
            x=var1,
            y=var2,
            color="surgical_lesion",
            trendline="ols"
            )
        st.plotly_chart(fig)

    #Si une variable est qualitative
    elif (data[var1].dtype == 'float64') | (data[var2].dtype == 'float64'):
        st.write("## Analyse bivariée : continue vs qualitative")
        fig = px.box(
            data,
            x=var1,
            y=var2,
            color="surgical_lesion"
            )
        st.plotly_chart(fig)

    #Si deux variables qualitatives
    else:
        st.write("## Analyse bivariée : continue vs qualitative")
        fig = px.histogram(
            data,
            x=var1,
            color=var2,
            barmode="group"
            )
        st.plotly_chart(fig)

with corTab:
    # Analyse multivariée
    st.write('## Analyse multivariée')
    # Heatmap
    correlation = data[list_columns].corr()
    fig = px.imshow(correlation,
                text_auto=".2f",
                color_continuous_scale = 'Viridis',
                range_color=(0, 1),
                title='Correlation Matrix',
                template='plotly_dark',
                width=800,
                height=800
            )
    fig.update_xaxes(tickangle=40)
    st.plotly_chart(fig)

with abTab:
    #Valeurs abérentes
    #Curseur pour le z-score
    zScore = st.slider(
        label = "Z-score",
        min_value = 0,
        max_value = 10,
        value = 3
        )

    #list columns execding zScore
    list_ab = []
    for col in data[list_columns].columns:
        # Si la variable est quantitative
        if (data[col].dtype == 'float64'):
            # si mean - zScore*std n'est pas entre le min and max
            if (data[col].mean() - zScore*data[col].std() > data[col].min()) | (data[col].mean() + zScore*data[col].std() < data[col].max()):
                list_ab.append(col)

    #Affiche la table avec les colonnes consernées 
    # avec les valeurs abérentes en rouge
    st.write('## Valeurs abérentes')
    st.write(
        data[list_ab].style.apply(
                lambda x: [
                "background-color: red; color: white; font-weight: bold" 
                if v > x.mean() + zScore*x.std() or v < x.mean() - zScore*x.std()
                else "" 
                for v in x],
                axis = 0
                )
            )


    #display the boxplot for each
    st.write('## Boxplots')
    for col in list_ab:
        fig = px.box(
            data,
            y=col,
            color="surgical_lesion"
            )
        st.plotly_chart(fig)


with missTab:
    na = data[list_columns].isna().sum()/data[list_columns].shape[0]*100
    fig = px.bar(na.sort_values(),
         title = "Pourcentage de valeurs manquantes par variable",
         color_discrete_sequence = ["#377eb8"],
         template='ggplot2',
         range_y = [0,100],
         width=800,
         labels={'value':'%', 'index':''},
         color=None
        ).update_traces(showlegend=False).update_xaxes(tickangle=45)
    st.plotly_chart(fig)

    fig2 = px.bar(data[list_columns].isna().sum(axis=1).value_counts(),
            title = "Nombre de valeurs manquantes par ligne",
            color_discrete_sequence = ["#377eb8"],
            template='ggplot2',
            range_y = [0,100],
            width=800,
            labels={'value':'%', 'index':''},
            color=None
            ).update_traces(showlegend=False).update_xaxes(tickangle=45)
    st.plotly_chart(fig2)

    #cursor for threshold of missing values for columns and rows
    thresholdRow = st.slider(
        label = "Retire les lignes avec plus de valeurs manquantes que :",
        key = 'thresholdRow',
        min_value = 0,
        max_value = data[list_columns].shape[1],
        value = 0
        )

    thresholdCol = st.slider(
        label = "Retire les colonnes avec plus de valeurs manquantes que :",
        key = 'thresholdCol',
        min_value = 0,
        max_value = 100,
        format = '%d%%',
        value = 50
        )

    #remove rows with more than thresholdRow missing values
    remainingData = data[list_columns].dropna(
        axis=0, thresh=thresholdRow
        ).dropna(
        axis=1, thresh=thresholdCol/100*data[list_columns].shape[0]
        )

    excludedRows = data[list_columns][~data[list_columns].index.isin(remainingData.index)]
    excludedCols = data[list_columns].columns[~data[list_columns].columns.isin(remainingData.columns)]

    #display the shape & table with remaining values
    st.write('## Valeurs manquantes')
    st.write(
        'Lignes retirés : ',
        data[list_columns].shape[0] - remainingData.shape[0],
        ' sur ',
        data[list_columns].shape[0]
        )
    st.write(
        'Colonnes retirés : ',
        data[list_columns].shape[1] - remainingData.shape[1],
        ' sur ',
        data[list_columns].shape[1]
      )
    st.write(remainingData)
    st.write('## Lignes retirées')
    st.write(excludedRows)
    st.write('## Colonnes retirées')
    st.write(excludedCols)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import string
from IPython import display  
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def createwells(x):

    import string
    row384 = list(string.ascii_uppercase[:16])
    col384 = [(f'{i:02d}') for i in range(1, 25, 1)]
    wells384 = []

    for r in row384:
        for c in col384:
            wells384.append(str(r+c))
    return(wells384 * x)


def plot_plates(df, featurename, vmax=None):
    output_dir = "platemaps"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = df.groupby(['Metadata_plate_map_name', 'Metadata_Well'])[featurename].mean().reset_index()

    batch = df['Metadata_plate_map_name'].unique().tolist()

    ncols = 2
    nrows = len(batch) // ncols + (len(batch) % ncols > 0)
    
    if vmax is None:
        vmax = df[featurename].max()
    
    fig = plt.figure(figsize=(ncols * 15, nrows * 10))

    for n, plate in enumerate(batch):                             
        ax = plt.subplot(nrows, ncols, n + 1)                     
        
        wells = df[df["Metadata_plate_map_name"] == plate]

        usedwells = wells['Metadata_Well'].tolist()
        allwells = createwells(1)
        fillup = set(allwells) - set(usedwells)

        fillplate = pd.DataFrame(fillup, columns=['Metadata_Well'])
        fillplate['Metadata_plate_map_name'] = plate
        fillplate[featurename] = 0  
        fillplate = fillplate[["Metadata_plate_map_name", "Metadata_Well", featurename]]
        
        dffull = pd.concat([fillplate, wells], axis=0)
        dffull['col'] = dffull.Metadata_Well.astype(str).str[1:3]
        dffull['row'] = dffull.Metadata_Well.astype(str).str[0]
        
        wells_pivot = dffull.pivot(columns="col", index="row", values=featurename)
        cmap = "OrRd"
        mask = wells_pivot.isnull()
        
        ax = sns.heatmap(wells_pivot, 
                         ax=ax,
                         vmin=0, vmax=vmax, 
                         square=True,
                         cmap=cmap,
                         annot=True,
                         linewidths=.8, linecolor='darkgray',
                         annot_kws={'fontsize': 8},
                         cbar_kws={'label': featurename, 'orientation': 'vertical', "shrink": .5},
                         mask=mask)
        ax.set_title(plate)

    plt.subplots_adjust(wspace=0.1, hspace=0.01)
    plt.suptitle(str(featurename), fontsize=50, y=0.92)

    plt.savefig(f"{output_dir}/PLATEMAP_{featurename}.pdf")
    plt.show()


def plot_images(df, condition):
    unique_compounds = df["Metadata_cmpdName"].unique()

    for compound in unique_compounds:
        compound_df = df[df["Metadata_cmpdName"] == compound]
        compound_df_grouped = compound_df.groupby("Metadata_plate_map_name")


        num_plates = len(compound_df_grouped)
        fig_width = num_plates * 6
        fig_height = 6
        figure, axarr = plt.subplots(1, num_plates, figsize=(fig_width, fig_height))

        if not isinstance(axarr, np.ndarray):
            axarr = np.array([axarr])

        for i, (plate_name, group) in enumerate(compound_df_grouped):
            sample_row = group.sample(1).iloc[0]
            image_path = sample_row["Filenames"]

            with Image.open(image_path) as img:
                axarr[i].set_title(f"{plate_name}", fontsize=12)
                axarr[i].imshow(img, cmap='jet')
                axarr[i].set_axis_off()

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.suptitle(f"Compound: {compound}, {condition}", fontsize=12)
        plt.savefig(f"{condition}_{compound}.pdf", format="pdf", bbox_inches="tight")
        plt.show()
        plt.close()


def run_pca_all_plates(df, color_column='Metadata_cmpdName', n_components=2, custom_palette=None, output_filename=None):
    combined_data = df.loc[:, ~df.columns.str.contains('Meta')].dropna()

    pca = PCA(n_components=n_components)
    X_pca_combined = pca.fit_transform(StandardScaler().fit_transform(combined_data))

    explained_variance = pca.explained_variance_ratio_
    pc1_var = explained_variance[0] * 100
    pc2_var = explained_variance[1] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    if custom_palette is None:
        unique_values = df[color_column].unique()
        colors = sns.color_palette('hls', len(unique_values))
        custom_palette = dict(zip(unique_values, colors))

    sns.scatterplot(
        x=X_pca_combined[:, 0], 
        y=X_pca_combined[:, 1], 
        hue=df[color_column].values, 
        palette=custom_palette,  
        s=20, 
        ax=ax1
    )

    ax1.set_title('PCA After Combining All Plates')
    ax1.set_xlabel(f'PC1 ({pc1_var:.2f}% var)')
    ax1.set_ylabel(f'PC2 ({pc2_var:.2f}% var)')

    display_scree_plot(pca, ax2)

    plt.tight_layout()
    
    if output_filename:
        fig.savefig(output_filename, dpi=300, format='png')

    plt.show()

def display_scree_plot(pca, ax):
    scree = pca.explained_variance_ratio_ * 100
    cumulative_variance = np.cumsum(scree)

    ax.bar(np.arange(len(scree)) + 1, scree, alpha=0.7, color='steelblue', label='Explained Variance')
    ax.plot(np.arange(len(scree)) + 1, cumulative_variance, color='red', marker='o', linestyle='--', linewidth=2, label='Cumulative Explained Variance')

    ax.set_xlabel('Number of Principal Components', fontsize=12)
    ax.set_ylabel('Explained Variance (%)', fontsize=12)
    ax.set_title('Scree Plot', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.set_xticks(np.arange(1, len(scree) + 1))
    ax.set_yticks(np.arange(0, 110, 10)) 
    ax.set_ylim(0, 105)

    ax.legend(loc='best', fontsize=10)


def run_pca_per_plate(df, color_column='Metadata_cmpdName', n_components=2, custom_palette=None, output_filename=None):
    pca_results = {}
    explained_variance_dict = {}

    unique_plates = df['Metadata_plate_map_name'].unique()
    n = len(unique_plates)
    nrows = int(np.ceil(n / 3.0))
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5 * nrows))
    axes = axes.flatten()

    if custom_palette is None:
        unique_values = df[color_column].unique()
        colors = sns.color_palette('hls', len(unique_values))
        custom_palette = dict(zip(unique_values, colors))

    for i, plate in enumerate(unique_plates):
        ax = axes[i]
        subset_df = df[df['Metadata_plate_map_name'] == plate]
        data_subset = subset_df.loc[:, ~subset_df.columns.str.contains('Meta')].dropna()
        x_subset = StandardScaler().fit_transform(data_subset)
        pca = PCA(n_components=n_components)
        X_pca_subset = pca.fit_transform(x_subset)

        explained_variance = pca.explained_variance_ratio_
        pc1_var = explained_variance[0] * 100
        pc2_var = explained_variance[1] * 100
        explained_variance_dict[plate] = (pc1_var, pc2_var)

        sns.scatterplot(
            x=X_pca_subset[:, 0], 
            y=X_pca_subset[:, 1], 
            hue=subset_df[color_column].values, 
            palette=custom_palette,  
            s=20, 
            ax=ax
        )

        ax.set_title(f'Plate: {plate}')
        ax.set_xlabel(f'PC1 ({pc1_var:.2f}% var)')
        ax.set_ylabel(f'PC2 ({pc2_var:.2f}% var)')

        pca_results[plate] = X_pca_subset

    for i in range(n, nrows * 3):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if output_filename:
        fig.savefig(output_filename, dpi=600, format='png')

    plt.show()


def group_features(feature_name):
    channels = ["ER", "DNA", "RNA", "AGP", "Mito"]
    
    if "AreaShape" in feature_name:
        if "Nuclei" in feature_name:
            return "AreaShape_Nuclei"
        elif "Cytoplasm" in feature_name:
            return "AreaShape_Cytoplasm"
        elif "Cells" in feature_name:
            return "AreaShape_Cells"

    for channel in channels:
        if channel in feature_name:
            if "Intensity" in feature_name:
                return f"{channel}_Intensity"
            elif "Granularity" in feature_name:            
                return f"{channel}_Granularity"
            elif "RadialDistribution" in feature_name:
                return f"{channel}_RadialDistribution"
        
    if "Neighbors" in feature_name:
        if "Cells" in feature_name:
            return "Neighbors_Cells"
        elif "Nuclei" in feature_name:
            return "Neighbors_nuclei"

    if "Children" in feature_name:
        if "Nuclei" in feature_name:
            return "Children_nuclei"

    if "Correlation" in feature_name:
        parts = feature_name.split('_')
        found_channels = [part for part in parts if part in channels]
        sorted_channels = sorted(found_channels)
        sorted_feature_name = '_'.join(['Correlation'] + sorted_channels)
        if len(sorted_channels) == 2:
            return f'correlation_{sorted_channels[0]}_{sorted_channels[1]}'
    return "Uncategorized"